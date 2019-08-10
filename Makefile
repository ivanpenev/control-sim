INCDIR = include
SRCDIR = src
OBJDIR = obj
LIBDIR = lib
BINDIR = bin
CC = icl
LD = link
CFLAGS = /Qstd=c99 /Qrestrict /Qopenmp /O3 /QxHost /DMKL_ILP64 \
		-I"%MKLROOT%"\include -I$(INCDIR)
LDFLAGS = /libpath:$(LIBDIR) controlsim.lib slicot.lib lpkaux.lib \
		mkl_intel_ilp64_dll.lib mkl_intel_thread_dll.lib mkl_core_dll.lib libiomp5md.lib
ARCH = lib
OBJFILES =  $(OBJDIR)\io.obj \
			$(OBJDIR)\random.obj \
			$(OBJDIR)\genlyap.obj \
			$(OBJDIR)\gencare.obj \
			$(OBJDIR)\mincost.obj 

$(OBJDIR)\io.obj: $(SRCDIR)\io.c
	$(CC) /c /Fo:$@ $(CFLAGS) $?

$(OBJDIR)\random.obj: $(SRCDIR)\random.c
	$(CC) /c /Fo:$@ $(CFLAGS) $?

$(OBJDIR)\genlyap.obj: $(SRCDIR)\genlyap.c
	$(CC) /c /Fo:$@ $(CFLAGS) $?

$(OBJDIR)\gencare.obj: $(SRCDIR)\gencare.c
	$(CC) /c /Fo:$@ $(CFLAGS) $?

$(OBJDIR)\mincost.obj: $(SRCDIR)\mincost.c
	$(CC) /c /Fo:$@ $(CFLAGS) $?

$(OBJDIR)\test_gencare.obj: $(SRCDIR)\test_gencare.c
	$(CC) /c /Fo:$@ $(CFLAGS) $?

controlsim.lib: $(OBJFILES)
	$(ARCH) /out:$(LIBDIR)\$@ $(OBJFILES)

test_gencare.exe: controlsim.lib $(OBJDIR)\test_gencare.obj
	$(LD) /out:$(BINDIR)\$@ $(OBJDIR)\test_gencare.obj $(LDFLAGS)