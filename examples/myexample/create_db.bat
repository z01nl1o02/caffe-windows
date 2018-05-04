@setlocal enabledelayedexpansion
::train.txt and test.txt should be in format as following
::classid\pic.jpg
::classid should be int
set cafferoot=C:\dev\online\github\dev-cs\caffe-windows\

!cafferoot!\Build\x64\Release\convert_imageset.exe --shuffle --resize_width=32 --resize_height=32 C:\dataset\cifar\split\train\ !cafferoot!\examples\myexample\train.txt !cafferoot!\examples\myexample\img_train_lmdb
!cafferoot!\Build\x64\Release\convert_imageset.exe --shuffle --resize_width=32 --resize_height=32 C:\dataset\cifar\split\test\ !cafferoot!\examples\myexample\test.txt !cafferoot!\examples\myexample\img_test_lmdb

@endlocal