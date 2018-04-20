@setlocal enabledelayedexpansion

set cafferoot=c:\dev\online\github\dev-happynear\caffe-windows

!cafferoot!\Build\x64\Release\compute_image_mean.exe  !cafferoot!\examples\myfile\img_train_lmdb !cafferoot!\examples\myfile\mean.binaryproto

@endlocal