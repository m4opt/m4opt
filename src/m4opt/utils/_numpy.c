#include <Python.h>
#include <numpy/arrayobject.h>

static npy_intp find_floor(const npy_intp *a, npy_intp x, npy_intp n) {
    npy_intp base = 0;
    while (n > 0) {
        npy_intp mid = base + n / 2;
        if (x < a[mid]) {
            n /= 2;
        } else {
            base = mid + 1;
            n -= n / 2 + 1;
        }
    }
    return base - 1;
}

#define SWAP(tp, a, b) do { \
    tp temp = a; \
    a = b; \
    b = temp; \
} while (0);

static npy_intp count_intersect1d(const npy_intp *a, const npy_intp *b, size_t n_a, size_t n_b) {
    const npy_intp *p = a, *q = b, *p_end = a + n_a, *q_end = b + n_b;
    npy_intp result = 0;
    while (1) {
        // Ensure that we are iterating over the shorter of the two arrays
        size_t n_p = p_end - p, n_q = q_end - q;
        if (n_p > n_q) {
            SWAP(const npy_intp *, p, q);
            SWAP(const npy_intp *, p_end, q_end);
            SWAP(size_t, n_p, n_q);
        }
        if (p == p_end)
        {
            // No more elements, done
            return result;
        }
        npy_intp hit = find_floor(q, *p, n_q);
        if (hit >= 0) {
            q += hit;
            if (*p == *q) {
                result++;
                q++;
            }
        }
        p++;
    }
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

static PyMethodDef methods[] = {
    {"count_intersect1d", (PyCFunction)py_count_intersect1d, METH_FASTCALL},
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
