// The test photo (375x250 RGB888, raw bytes from photo.rgb), embedded with .incbin: no C array.
    .section .rodata
    .global kPhotoData
    .global kPhotoDataEnd
    .balign 16
kPhotoData:
    .incbin "photo.rgb"
kPhotoDataEnd:
    .previous
