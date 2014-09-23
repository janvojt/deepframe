CC=g++ # This is the main compiler
#CC=clang --analyze # and comment out the linker last line for sanity

SRCDIR=src/main
BUILDDIR=build/main
TARGETDIR=bin
TARGET=$(TARGETDIR)/xoraan

TESTDIR=src/test
TESTBUILDDIR=build/test
TESTTARGET=$(TARGETDIR)/xoraan-test
 
SRCEXT=cpp
SOURCES=$(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
TESTSOURCES=$(shell find $(TESTDIR) -type f -name *.$(SRCEXT))
#SOURCEDIRS=$(shell find $(SRCDIR) -type d)
OBJECTS=$(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
TESTOBJECTS=$(patsubst $(TESTDIR)/%,$(TESTBUILDDIR)/%,$(TESTSOURCES:.$(SRCEXT)=.o))

CFLAGS=-g # -Wall
LIB=-pthread -llog4cpp -lgtest -lgtest_main
INC=-I include
TESTINC=-Isrc/test

all: $(TARGET) $(TESTTARGET)

$(TARGET): $(OBJECTS)
	@echo "Linking..."
	@mkdir -p $(TARGETDIR)
	$(CC) $^ -o $(TARGET) $(LIB)

$(TESTTARGET): $(TESTOBJECTS)
	@echo "Linking tests..."
	@mkdir -p $(TARGETDIR)
	$(CC) $^ -o $(TESTTARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo "Building..."
	@mkdir -p $(BUILDDIR) $(shell dirname $@)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(TESTBUILDDIR)/%.o: $(TESTDIR)/%.$(SRCEXT)
	@echo "Building tests..."
	@mkdir -p $(TESTBUILDDIR) $(shell dirname $@)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	$(RM) -r $(BUILDDIR) $(TARGET) $(TESTBUILDDIR) $(TESTTARGET)

