CC=g++ # This is the main compiler
#CC=clang --analyze # and comment out the linker last line for sanity
SRCDIR=src
BUILDDIR=build
TARGETDIR=bin
TARGET=$(TARGETDIR)/xoraan
 
SRCEXT=cpp
SOURCES=$(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
SOURCEDIRS=$(shell find $(SRCDIR) -type d)
OBJECTS=$(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS=-g # -Wall
LIB=-pthread -llog4cpp
INC=-I include

$(TARGET): $(OBJECTS)
	@echo "Linking..."
	@mkdir -p $(TARGETDIR)
	@echo " $(CC) $^ -o $(TARGET) $(LIB)"
	$(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo "Building..."
	@mkdir -p $(BUILDDIR) $(shell dirname $@)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning..."; 
	@echo " $(RM) -r $(BUILDDIR) $(TARGET)"
	$(RM) -r $(BUILDDIR) $(TARGET)

