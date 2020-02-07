set(vtkViewsQt_LOADED 1)
set(vtkViewsQt_DEPENDS "vtkGUISupportQt;vtkViewsInfovis")
set(vtkViewsQt_LIBRARIES "vtkViewsQt")
set(vtkViewsQt_INCLUDE_DIRS "${VTK_INSTALL_PREFIX}/include/vtk-6.1")
set(vtkViewsQt_LIBRARY_DIRS "")
set(vtkViewsQt_RUNTIME_LIBRARY_DIRS "${VTK_INSTALL_PREFIX}/lib")
set(vtkViewsQt_WRAP_HIERARCHY_FILE "${CMAKE_CURRENT_LIST_DIR}/vtkViewsQtHierarchy.txt")
set(vtkViewsQt_EXCLUDE_FROM_WRAPPING 1)

