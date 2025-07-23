#include <math.h>
#include <stdio.h>

int square(int i) {
	return i * i;
}

int NI_Correlate1D(PyArrayObject *input, PyArrayObject *weights,
                   int axis, PyArrayObject *output, NI_ExtendMode mode,
                   double cval, npy_intp origin)
{
    FILE *fptr;

    // Open a file in writing mode
    fptr = fopen("/home/chamomile/filename.txt", "wa");

    int symmetric = 0, more;
    npy_intp ii, jj, ll, lines, length, size1, size2, filter_size;
    double *ibuffer = NULL, *obuffer = NULL;
    npy_double *fw;
    NI_LineBuffer iline_buffer, oline_buffer;
    NPY_BEGIN_THREADS_DEF;

    /* test for symmetry or anti-symmetry: */
    filter_size = PyArray_SIZE(weights);
    size1 = filter_size / 2;
    size2 = filter_size - size1 - 1;
    fw = (void *)PyArray_DATA(weights);

    if (filter_size & 0x1) {
        symmetric = 1;
        for(ii = 1; ii <= filter_size / 2; ii++) {
            if (fabs(fw[ii + size1] - fw[size1 - ii]) > DBL_EPSILON) {
                symmetric = 0;
                break;
            }
        }
        if (symmetric == 0) {
            symmetric = -1;
            for(ii = 1; ii <= filter_size / 2; ii++) {
                if (fabs(fw[size1 + ii] + fw[size1 - ii]) > DBL_EPSILON) {
                    symmetric = 0;
                    break;
                }
            }
        }
    }
    /* allocate and initialize the line buffers: */
    lines = -1;
    if (!NI_AllocateLineBuffer(input, axis, size1 + origin, size2 - origin,
                                                         &lines, BUFFER_SIZE, &ibuffer))
        goto exit;
    if (!NI_AllocateLineBuffer(output, axis, 0, 0, &lines, BUFFER_SIZE,
                                                         &obuffer))
        goto exit;
    if (!NI_InitLineBuffer(input, axis, size1 + origin, size2 - origin,
                                                            lines, ibuffer, mode, cval, &iline_buffer))
        goto exit;
    if (!NI_InitLineBuffer(output, axis, 0, 0, lines, obuffer, mode, 0.0,
                                                 &oline_buffer))
        goto exit;

    NPY_BEGIN_THREADS;
    length = PyArray_NDIM(input) > 0 ? PyArray_DIM(input, axis) : 1;
    fw += size1;
    char str[300];
    //sprintf(str, "symmetric %d size1 %ld size2 %ld \n", symmetric, size1, size2);
    //printf(str);
    // Write some text to the file
    //fprintf(fptr, "str");

    // Close the file
    /* iterate over all the array lines: */
    do {
        /* copy lines from array to buffer: */
        if (!NI_ArrayToLineBuffer(&iline_buffer, &lines, &more)) {
            goto exit;
        }
        /* iterate over the lines in the buffers: */
        for(ii = 0; ii < lines; ii++) {
            /* get lines: */
            double *iline = NI_GET_LINE(iline_buffer, ii) + size1;
            double *oline = NI_GET_LINE(oline_buffer, ii);

            //sprintf(str, "Printing iline oline %d %f %f %f %f \n", ii, iline, *oline, iline_buffer, oline_buffer);
            //printf(str);

            /* the correlation calculation: */
            if (symmetric > 0) {
                for(ll = 0; ll < length; ll++) {
                    oline[ll] = iline[0] * fw[0];
                    for(jj = -size1 ; jj < 0; jj++){
                        oline[ll] += (iline[jj] + iline[-jj]) * fw[jj];
                        //printf("jj %ld -jj %ld ll %ld \n", jj, -jj, ll);

                        //printf("iline[jj] %E iline[-jj] %E fw[jj] %E iline[0] %E fw[0] %E \n", iline[jj], iline[-jj], fw[jj], iline[0], fw[0]);
                        }
                    //printf("oline[ll] %E \n", oline[ll]);
                    ++iline;
                }
            } else if (symmetric < 0) {
                for(ll = 0; ll < length; ll++) {
                    oline[ll] = iline[0] * fw[0];
                    for(jj = -size1 ; jj < 0; jj++)
                        oline[ll] += (iline[jj] - iline[-jj]) * fw[jj];
                        //printf("jj %ld -jj %ld ll %ld \n", jj, -jj, ll);
                        //printf("oline[ll] %f iline[jj] %f iline[-jj] %E \n", oline[ll], iline[jj], iline[-jj]);
                    ++iline;
                }
            } else {
                for(ll = 0; ll < length; ll++) {
                    oline[ll] = iline[size2] * fw[size2];
                    for(jj = -size1; jj < size2; jj++) {
                        oline[ll] += iline[jj] * fw[jj];
                        //printf("jj %ld \n", jj);
                        //printf("jj %ld -jj %ld ll %ld \n", jj, -jj, ll);
                        //printf("jj %ld iline[jj] %E fw[jj] %E iline[size2] %E fw[size2] %E \n", jj, iline[jj], fw[jj], iline[size2], fw[size2]);
                    }
                    //printf("oline[ll] %E \n", oline[ll]);
                    ++iline;
                }
            }
        //printf("Printing iline oline %ld %f %f \n", ii, *iline, oline[1]);

        }
        /* copy lines from buffer to array: */
        if (!NI_LineBufferToArray(&oline_buffer)) {
            goto exit;
        }
    } while(more);

exit:
    NPY_END_THREADS;
    free(ibuffer);
    free(obuffer);
    fclose(fptr);
    return PyErr_Occurred() ? 0 : 1;
}