# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build

# Include any dependencies generated for this target.
include CMakeFiles/emosqpstatic.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/emosqpstatic.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/emosqpstatic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/emosqpstatic.dir/flags.make

CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.o: ../src/osqp/auxil.c
CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/auxil.c

CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/auxil.c > CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/auxil.c -o CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/error.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/error.c.o: ../src/osqp/error.c
CMakeFiles/emosqpstatic.dir/src/osqp/error.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/error.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/error.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/error.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/error.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/error.c

CMakeFiles/emosqpstatic.dir/src/osqp/error.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/error.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/error.c > CMakeFiles/emosqpstatic.dir/src/osqp/error.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/error.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/error.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/error.c -o CMakeFiles/emosqpstatic.dir/src/osqp/error.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.o: ../src/osqp/lin_alg.c
CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/lin_alg.c

CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/lin_alg.c > CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/lin_alg.c -o CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.o: ../src/osqp/osqp.c
CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/osqp.c

CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/osqp.c > CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/osqp.c -o CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.o: ../src/osqp/proj.c
CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/proj.c

CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/proj.c > CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/proj.c -o CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.o: ../src/osqp/scaling.c
CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/scaling.c

CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/scaling.c > CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/scaling.c -o CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/util.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/util.c.o: ../src/osqp/util.c
CMakeFiles/emosqpstatic.dir/src/osqp/util.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/util.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/util.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/util.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/util.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/util.c

CMakeFiles/emosqpstatic.dir/src/osqp/util.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/util.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/util.c > CMakeFiles/emosqpstatic.dir/src/osqp/util.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/util.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/util.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/util.c -o CMakeFiles/emosqpstatic.dir/src/osqp/util.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.o: ../src/osqp/kkt.c
CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/kkt.c

CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/kkt.c > CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/kkt.c -o CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.o: ../src/osqp/workspace.c
CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/workspace.c

CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/workspace.c > CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/workspace.c -o CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.o: ../src/osqp/qdldl.c
CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/qdldl.c

CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/qdldl.c > CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/qdldl.c -o CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.s

CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.o: CMakeFiles/emosqpstatic.dir/flags.make
CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.o: ../src/osqp/qdldl_interface.c
CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.o: CMakeFiles/emosqpstatic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building C object CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.o -MF CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.o.d -o CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.o -c /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/qdldl_interface.c

CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/qdldl_interface.c > CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.i

CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/src/osqp/qdldl_interface.c -o CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.s

# Object files for target emosqpstatic
emosqpstatic_OBJECTS = \
"CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/error.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/util.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.o" \
"CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.o"

# External object files for target emosqpstatic
emosqpstatic_EXTERNAL_OBJECTS =

out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/auxil.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/error.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/lin_alg.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/osqp.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/proj.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/scaling.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/util.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/kkt.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/workspace.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/qdldl.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/src/osqp/qdldl_interface.c.o
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/build.make
out/libemosqpstatic.a: CMakeFiles/emosqpstatic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Linking C static library out/libemosqpstatic.a"
	$(CMAKE_COMMAND) -P CMakeFiles/emosqpstatic.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/emosqpstatic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/emosqpstatic.dir/build: out/libemosqpstatic.a
.PHONY : CMakeFiles/emosqpstatic.dir/build

CMakeFiles/emosqpstatic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/emosqpstatic.dir/cmake_clean.cmake
.PHONY : CMakeFiles/emosqpstatic.dir/clean

CMakeFiles/emosqpstatic.dir/depend:
	cd /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30 /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30 /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build /Users/carlaxelfolkestad/Coding_projects/koopman-flight-experiments/koopman_learning_and_control/koopman_core/controllers/embedded_controllers/nmpc_12_4_30/build/CMakeFiles/emosqpstatic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/emosqpstatic.dir/depend

