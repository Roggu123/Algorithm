# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/Build

# Include any dependencies generated for this target.
include CMakeFiles/AgeGender.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/AgeGender.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/AgeGender.dir/flags.make

CMakeFiles/AgeGender.dir/AgeGender.cpp.o: CMakeFiles/AgeGender.dir/flags.make
CMakeFiles/AgeGender.dir/AgeGender.cpp.o: ../AgeGender.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/AgeGender.dir/AgeGender.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/AgeGender.dir/AgeGender.cpp.o -c /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/AgeGender.cpp

CMakeFiles/AgeGender.dir/AgeGender.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/AgeGender.dir/AgeGender.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/AgeGender.cpp > CMakeFiles/AgeGender.dir/AgeGender.cpp.i

CMakeFiles/AgeGender.dir/AgeGender.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/AgeGender.dir/AgeGender.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/AgeGender.cpp -o CMakeFiles/AgeGender.dir/AgeGender.cpp.s

CMakeFiles/AgeGender.dir/AgeGender.cpp.o.requires:

.PHONY : CMakeFiles/AgeGender.dir/AgeGender.cpp.o.requires

CMakeFiles/AgeGender.dir/AgeGender.cpp.o.provides: CMakeFiles/AgeGender.dir/AgeGender.cpp.o.requires
	$(MAKE) -f CMakeFiles/AgeGender.dir/build.make CMakeFiles/AgeGender.dir/AgeGender.cpp.o.provides.build
.PHONY : CMakeFiles/AgeGender.dir/AgeGender.cpp.o.provides

CMakeFiles/AgeGender.dir/AgeGender.cpp.o.provides.build: CMakeFiles/AgeGender.dir/AgeGender.cpp.o


# Object files for target AgeGender
AgeGender_OBJECTS = \
"CMakeFiles/AgeGender.dir/AgeGender.cpp.o"

# External object files for target AgeGender
AgeGender_EXTERNAL_OBJECTS =

AgeGender: CMakeFiles/AgeGender.dir/AgeGender.cpp.o
AgeGender: CMakeFiles/AgeGender.dir/build.make
AgeGender: /usr/local/lib/libopencv_stitching.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_superres.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_videostab.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_aruco.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_bgsegm.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_bioinspired.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_ccalib.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_dnn_objdetect.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_dpm.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_face.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_fuzzy.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_hfs.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_img_hash.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_line_descriptor.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_optflow.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_reg.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_rgbd.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_saliency.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_stereo.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_structured_light.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_surface_matching.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_tracking.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_xfeatures2d.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_ximgproc.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_xobjdetect.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_xphoto.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_shape.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_photo.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_datasets.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_plot.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_text.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_dnn.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_ml.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_video.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_calib3d.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_features2d.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_highgui.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_videoio.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_phase_unwrapping.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_flann.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_imgcodecs.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_objdetect.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_imgproc.3.4.1.dylib
AgeGender: /usr/local/lib/libopencv_core.3.4.1.dylib
AgeGender: CMakeFiles/AgeGender.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable AgeGender"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/AgeGender.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/AgeGender.dir/build: AgeGender

.PHONY : CMakeFiles/AgeGender.dir/build

CMakeFiles/AgeGender.dir/requires: CMakeFiles/AgeGender.dir/AgeGender.cpp.o.requires

.PHONY : CMakeFiles/AgeGender.dir/requires

CMakeFiles/AgeGender.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/AgeGender.dir/cmake_clean.cmake
.PHONY : CMakeFiles/AgeGender.dir/clean

CMakeFiles/AgeGender.dir/depend:
	cd /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/Build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/Build /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/Build /Users/ruogulu/Desktop/Study/Algorithm/Practice/CV/FaceDetection/age_gender/Build/CMakeFiles/AgeGender.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/AgeGender.dir/depend

