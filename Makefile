#load config.mk
-include config.mk
SETUP ?= kh
EOS ?= adiabatic
FLUX ?= HLLC 
RC_ORDER ?= 2
USE_SINGLE_PRECISION ?= 0
GAUKUK_DENSITY_FLOOR ?= 1e-16
GAUKUK_PRESSURE_FLOOR ?= 1e-16

CXX = nvcc 

# select precision flag
ifeq ($(USE_SINGLE_PRECISION),1)
    PRECISION_FLAG = -DGAUKUK_USE_FLOAT
else
    PRECISION_FLAG =
endif

# select reconstruction flag
ifeq ($(FLUX), ROE)
    FLUX_FLAG = -DUSE_ROE
else
    FLUX_FLAG = -DUSE_HLLC
endif

# select reconstruction flag
ifeq ($(RC_ORDER),1)
    RC_FLAG = -DUSE_FIRST_ORDER
else
    RC_FLAG = -DUSE_PLM
endif

CXXFLAGS = 

CXXFLAGS += $(PRECISION_FLAG)
CXXFLAGS += -DGAUKUK_DENSITY_FLOOR=$(GAUKUK_DENSITY_FLOOR) \
            -DGAUKUK_PRESSURE_FLOOR=$(GAUKUK_PRESSURE_FLOOR) \
			$(FLUX_FLAG)
CXXFLAGS += $(RC_FLAG)
INCLUDES = 
LDFLAGS = 

# paths 
MAIN_DIR = ./src
OBJ_DIR = ./obj
BIN_DIR = ./bin
TARGET = $(BIN_DIR)/gaukuk.sim

# obtain source files (.cpp files)
CPP_SRCS = $(MAIN_DIR)/main.cpp \
	   	$(MAIN_DIR)/sim.cpp \
	   	$(MAIN_DIR)/setup/setup_$(SETUP).cpp \
		$(MAIN_DIR)/boundary/boundary.cpp \
		$(MAIN_DIR)/evolution/forward_euler.cpp \
		$(MAIN_DIR)/evolution/rk2.cpp \
		$(MAIN_DIR)/evolution/rk3.cpp \
	   	$(MAIN_DIR)/utils/write_sim.cpp \
	   	$(MAIN_DIR)/utils/read_config.cpp 

CU_SRCS = $(MAIN_DIR)/eos/$(EOS).cu \
		  $(MAIN_DIR)/boundary/outflow.cu \
		  $(MAIN_DIR)/boundary/periodic.cu \
		  $(MAIN_DIR)/evolution/update_cons.cu 

# create object files (.o 文件)
CPP_OBJS = $(CPP_SRCS:$(MAIN_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CU_OBJS = $(CU_SRCS:$(MAIN_DIR)/%.cu=$(OBJ_DIR)/%.o)
OBJS = $(CPP_OBJS) $(CU_OBJS)


# target
all: $(TARGET)

# compile rules
$(OBJ_DIR)/%.o: $(MAIN_DIR)/%.cpp | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# chain rule
$(TARGET): $(OBJS) | $(BIN_DIR)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJS) $(LDFLAGS) -o $(TARGET)

# clean 
clean:
	rm -rf $(OBJ_DIR)/* $(TARGET)

$(OBJ_DIR)/%.o: $(MAIN_DIR)/%.cu | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
# create obj directory (if not exist)
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# create bin directory (if not exist)
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

.PHONY: all clean