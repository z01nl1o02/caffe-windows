@setlocal enabledelayedexpansion

set cafferoot=c:\dev\online\github\dev-happynear\caffe-windows

!cafferoot!\Build\x64\Release\caffe.exe train -solver=!cafferoot!\examples\myfile\solver.prototxt

@endlocal