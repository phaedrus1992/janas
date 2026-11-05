/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * npu_ir.h - a small writer of OpenVINO IR (version 11): the XML of a graph
 * and its constants' bytes, the input of the NPU's compiler. Private to
 * npu.c.
 *
 * Layers have one output each; a layer's id is its number, its output
 * port's id its number of inputs, as the IR wants. Element types are IR's
 * names: "f32", "f16", "i8", "i64".
 */
#ifndef JANAS_COMMON_NPU_IR_H
#define JANAS_COMMON_NPU_IR_H

#include <stddef.h>
#include <stdint.h>

#define NPU_IR_LAYERS 160
#define NPU_IR_DIMS 4

struct npu_ir {
    char *xml;
    size_t len, cap;
    uint8_t *bin;
    size_t nbin;
    int nl, ne, failed, nres;
    int nd[NPU_IR_LAYERS], dims[NPU_IR_LAYERS][NPU_IR_DIMS], ins[NPU_IR_LAYERS];
    const char *et[NPU_IR_LAYERS];
    int edge[NPU_IR_LAYERS * 2][3]; /* from, to, to's port */
};

void npu_ir_begin(struct npu_ir *g, const char *name);
int npu_ir_param(struct npu_ir *g, const char *name, const char *et, int nd,
                 const int *dims);
/* bytes copied into the constants */
int npu_ir_const(struct npu_ir *g, const char *name, const char *et, int nd,
                 const int *dims, const void *p, size_t bytes);
int npu_ir_convert(struct npu_ir *g, const char *name, int a, const char *et);
int npu_ir_multiply(struct npu_ir *g, const char *name, int a, int b);
int npu_ir_add(struct npu_ir *g, const char *name, int a, int b);
/* an element-wise operation of two inputs, numpy broadcast, of opset1:
   "Add", "Subtract", "Multiply", "Divide" (output dims: the larger of
   each) */
int npu_ir_binary(struct npu_ir *g, const char *name, const char *type, int a,
                  int b);
/* the mean along one axis, kept as a dimension of 1 */
int npu_ir_reduce_mean(struct npu_ir *g, const char *name, int a, int axis);
/* any reduction of opset1 along one axis, kept: "ReduceMean", "ReduceMax" */
int npu_ir_reduce(struct npu_ir *g, const char *name, const char *type, int a,
                  int axis);
/* an element-wise function of opset1: "Sigmoid", "Tanh", "Sqrt", "Abs" */
int npu_ir_unary(struct npu_ir *g, const char *name, const char *type, int a);
/* a's values on levels steps between lo and hi (opset1 FakeQuantize): how
   OpenVINO asks for integer arithmetic */
int npu_ir_fake_quantize(struct npu_ir *g, const char *name, int a, float lo,
                         float hi, int levels);
/* a constant of one FP16 number, broadcast against anything */
int npu_ir_scalar(struct npu_ir *g, const char *name, float v);
/* a x b^T for 2-D a (m x k) and b (n x k) */
int npu_ir_matmul_tb(struct npu_ir *g, const char *name, int a, int b,
                     const char *names);
void npu_ir_result(struct npu_ir *g, int a);
/* ---- composites ---- */

/* weights of rows x cols in INT8 with an FP16 scale a row, as FP16 */
int npu_ir_weights_i8(struct npu_ir *g, const char *name, const int8_t *q,
                      const uint16_t *s, int rows, int cols);
/* x (m x d, f32) normalized by its root mean square, times w (d floats),
   in f32: the sum of squares of a residual with outliers in the tens of
   thousands stays finite (tested to 30000, 20 Oct 2026) */
int npu_ir_rms_norm(struct npu_ir *g, const char *name, int x, const float *w,
                    int d, float eps);
/* down (act(x gate^T) * (x up^T)), x FP16 m x d, the three matrices from
   npu_ir_weights_i8 (gate and up ff x d, down d x ff); act 0 SiLU, 1 the
   tanh form of GELU; FP16 out (m x d) */
int npu_ir_ffn(struct npu_ir *g, const char *name, int x, int gate, int up,
               int down, int act);

/* closes the XML; 0, or -1 if any step failed (out of memory or layers) */
int npu_ir_end(struct npu_ir *g);
void npu_ir_free(struct npu_ir *g);

/*
 * The compiler's input: its version, the number of parts (2), then the
 * XML's and the constants' sizes and bytes. Malloc'd, NULL on failure.
 */
uint8_t *npu_ir_model(const struct npu_ir *g, uint16_t major, uint16_t minor,
                      size_t *bytes);

#endif
