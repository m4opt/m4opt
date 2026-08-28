#include <Python.h>
#include <numpy/arrayobject.h>

/*
 * count_intersect1d is based on the one-sided lower_bound strategy used by
 * LLVM/libc++ for std::set_intersection with forward iterators.
 *
 * Basic approach:
 * 1) For a current value from one sorted array, search the other array using
 *    a one-sided (galloping/exponential) lower_bound.
 * 2) Swap roles and do the symmetric step from the other array.
 * 3) If both cursors land on the same value, count one intersection element
 *    and advance both pointers.
 *
 * This pure-C implementation was created with assistance from AI:
 * OpenAI ChatGPT gpt-5.3-codex, via Continue.
 */

static const npy_intp *lower_bound_bisecting(
    const npy_intp *first,
    npy_intp value,
    size_t len)
{
    while (len != 0) {
        const size_t half = len >> 1;
        const npy_intp *mid = first + half;
        if (*mid < value) {
            first = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    return first;
}

/*
 * One-sided lower_bound (aka meta binary search):
 * - grow a probe distance exponentially to bracket the target region,
 * - then run a short binary search inside that region.
 */
static const npy_intp *lower_bound_onesided(
    const npy_intp *first,
    const npy_intp *last,
    npy_intp value)
{
    if (first == last || !(*first < value))
        return first;

    size_t step = 1;
    while (first != last) {
        const size_t remaining = (size_t)(last - first);
        if (step >= remaining)
            return lower_bound_bisecting(first, value, remaining);

        const npy_intp *it = first + step;
        if (!(*it < value)) {
            if (step == 1)
                return it;
            return lower_bound_bisecting(first, value, step);
        }

        first = it;
        if (step > (remaining >> 1))
            step = remaining;
        else
            step <<= 1;
    }

    return first;
}

static npy_intp count_intersect1d(
    const npy_intp *a,
    const npy_intp *b,
    size_t n_a,
    size_t n_b)
{
    const npy_intp *first1 = a, *last1 = a + n_a;
    const npy_intp *first2 = b, *last2 = b + n_b;
    npy_intp result = 0;

    while (first1 != last1 && first2 != last2) {
        first1 = lower_bound_onesided(first1, last1, *first2);
        if (first1 == last1)
            break;

        first2 = lower_bound_onesided(first2, last2, *first1);
        if (first2 == last2)
            break;

        if (*first1 == *first2) {
            ++result;
            ++first1;
            ++first2;
        }
    }

    return result;
}

static PyObject *py_count_intersect1d(PyObject *NPY_UNUSED(self), PyObject *const *args, Py_ssize_t nargs) {
    PyObject *a = NULL, *b = NULL, *result = NULL;
    if (nargs != 2) {
        PyErr_Format(PyExc_TypeError, "count_intersect1d() takes exactly two arguments (%zd given)", (ssize_t) nargs);
        goto done;
    }
    if (!(a = PyArray_FROMANY(args[0], NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO))) goto done;
    if (!(b = PyArray_FROMANY(args[1], NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO))) goto done;
    result = PyLong_FromSsize_t(count_intersect1d(
        (const npy_intp *) PyArray_DATA((PyArrayObject *) a),
        (const npy_intp *) PyArray_DATA((PyArrayObject *) b),
        PyArray_SIZE((PyArrayObject *) a),
        PyArray_SIZE((PyArrayObject *) b)
    ));
done:
    Py_XDECREF(a);
    Py_XDECREF(b);
    return result;
}

static PyObject *py_count_intersect1d_combinations(PyObject *NPY_UNUSED(self), PyObject *arg) {
    Py_ssize_t num_arrays = PySequence_Length(arg);
    if (num_arrays < 0) return NULL;
    if (num_arrays < 2)
    {
        PyErr_Format(
            PyExc_ValueError,
            "count_intersect1d_combinations() expects a sequence of at least 2 arrays (%zd given)",
            (ssize_t) num_arrays
        );
        return NULL;
    }

    PyObject *result = NULL;
    Py_ssize_t result_len = num_arrays * (num_arrays - 1) / 2;
    PyObject **arrays = PyMem_Malloc(sizeof(PyObject *) * num_arrays);
    const npy_intp **data = PyMem_Malloc(sizeof(const npy_intp *) * num_arrays);
    npy_intp *n = PyMem_Malloc(sizeof(npy_intp) * num_arrays);
    Py_ssize_t *is = PyMem_Malloc(sizeof(Py_ssize_t) * result_len);
    Py_ssize_t *js = PyMem_Malloc(sizeof(Py_ssize_t) * result_len);
    if (!(arrays && data && n))
    {
        PyErr_NoMemory();
        goto free_arrays;
    }

    // Get pointers to the Numpy data for each input array.
    // It's important to do this here rather than in the computation loop below
    // so that the Python overhead grow like O(N) rather than O(N^2) in the
    // number of arrays N.
    for (Py_ssize_t i = 0; i < num_arrays; i ++) arrays[i] = NULL;
    for (Py_ssize_t i = 0; i < num_arrays; i ++)
    {
        PyObject *item = PySequence_GetItem(arg, i);
        if (!item) goto decref_objects;
        PyObject *array = PyArray_FROMANY(item, NPY_INTP, 1, 1, NPY_ARRAY_CARRAY_RO);
        Py_DECREF(item);
        if (!array) goto decref_objects;
        arrays[i] = array;
        data[i] = (const npy_intp *) PyArray_DATA((PyArrayObject *) array);
        n[i] = PyArray_SIZE((PyArrayObject *) array);
    }

    // Allocate output array. If there are N input arrays, then the output
    // array size is given by the binomial coefficient:
    //
    //      ⎛ N ⎞
    //      ⎜   ⎟
    //      ⎝ 2 ⎠
    //
    if (!(result = PyArray_SimpleNew(1, &result_len, NPY_INTP))) goto decref_objects;
    npy_intp *result_data = PyArray_DATA((PyArrayObject *) result);

    Py_BEGIN_ALLOW_THREADS;
    // Build lookup table from output index to input indices.
    {
        Py_ssize_t result_i = 0;
        for (Py_ssize_t i = 0; i < num_arrays; i ++)
        {
            for (Py_ssize_t j = i + 1; j < num_arrays; j ++)
            {
                is[result_i] = i;
                js[result_i] = j;
                result_i++;
            }
        }
    }
    // The lookup tables ``is`` and ``js`` defined above allow us to correctly
    // parallelize the following loop, which otherwise would only be correct
    // when executed serially because of the order-dependent increment of ``k``
    // inside the loop body. It also results in more efficient parallelism
    // because the work per loop is more balanced.
    //
    //      // incorrect, inefficient
    //      Py_ssize_t k = 0;
    //      #pragma omp parallel for
    //      for (Py_ssize_t i = 0; i < num_arrays; i ++)
    //          for (Py_ssize_t j = i + 1; j < num_arrays; j ++)
    //              result_data[k++] = count_intersect1d(data[i], data[j], n[i], n[j]);
    //
    #pragma omp parallel for
    for (Py_ssize_t result_i = 0; result_i < result_len; result_i ++)
    {
        Py_ssize_t i = is[result_i], j = js[result_i];
        result_data[result_i] = count_intersect1d(data[i], data[j], n[i], n[j]);
    }
    Py_END_ALLOW_THREADS;

decref_objects:
    for (Py_ssize_t i = 0; i < num_arrays; i++)
        Py_XDECREF(arrays[i]);
free_arrays:
    PyMem_Free(arrays);
    PyMem_Free(data);
    PyMem_Free(n);
    PyMem_Free(is);
    PyMem_Free(js);
    return result;
}

static PyMethodDef methods[] = {
    {"count_intersect1d", (PyCFunction)py_count_intersect1d, METH_FASTCALL},
    {"count_intersect1d_combinations", (PyCFunction)py_count_intersect1d_combinations, METH_O},
    {/* Sentinel */}
};

static PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_numpy",
    NULL, 0, methods
};

PyMODINIT_FUNC
PyInit__numpy(void)
{
    import_array();
    return PyModuleDef_Init(&moduledef);
}
