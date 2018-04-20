@setlocal enabledelayedexpansion

set cafferoot=c:\dev\online\github\dev-happynear\caffe-windows

!cafferoot!\Build\x64\Release\convert_imageset.exe --shuffle --resize_width=256 --resize_height=256 !cafferoot!\data\re\ !cafferoot!\examples\myfile\train.txt !cafferoot!\examples\myfile\img_train_lmdb
!cafferoot!\Build\x64\Release\convert_imageset.exe --shuffle --resize_width=256 --resize_height=256 !cafferoot!\data\re\ !cafferoot!\examples\myfile\test.txt !cafferoot!\examples\myfile\img_test_lmdb

@endlocal