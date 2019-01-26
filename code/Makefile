SRC_COM = ops.cc conv2d.cc matmul.cc cnn.cc pool2d.cc activation.cc batchnorm.cc concat.cc split.cc noop.cc element.cc substitution.cc subst_examples.cc graph_to_trt.cc
SRC_GPU = $(SRC_COM) $(wildcard *_cudnn.cu) cuda_helper.cu
SRC_TRT = $(SRC_GPU)
SRC_CPU = $(SRC_COM) $(wildcard *_mkl.cc)

CXXFLAGS_COM = -std=c++11
CXXFLAGS_GPU = $(CXXFLAGS_COM) -DUSE_CUDNN -O2 -arch=compute_70 -code=sm_70
CXXFLAGS_CPU = $(CXXFLAGS_COM) -DUSE_MKL -O3 -march=native -I$(MKLPATH)/include
CXXFLAGS_TRT = $(CXXFLAGS_GPU) -DTRT -I$(TRTPATH)/include

LDFLAGS_GPU = -lcudnn -lcublas
LDFLAGS_CPU = -L$(MKLPATH)/lib -lmklml_intel -liomp5
LDFLAGS_TRT = $(LDFLAGS_GPU) -L$(TRTPATH)/lib -lnvinfer

BIN_GPU = ../mf
BIN_CPU = ../mf_mkl
BIN_TRT = ../mf_trt

gpu: $(BIN_GPU)

cpu: $(BIN_CPU)

trt: $(BIN_TRT)

$(BIN_GPU): $(SRC_GPU)
		nvcc -o $@ $(CXXFLAGS_GPU) $^ $(LDFLAGS_GPU)

$(BIN_CPU): $(SRC_CPU)
		$(CXX) -o $@ $(CXXFLAGS_CPU) $^ $(LDFLAGS_CPU)

$(BIN_TRT): $(SRC_TRT)
		nvcc -o $@ $(CXXFLAGS_TRT) $^ $(LDFLAGS_TRT)

clean:
		$(RM) $(BIN_GPU) $(BIN_CPU) $(BIN_TRT)

.PHONY: gpu cpu trt clean

