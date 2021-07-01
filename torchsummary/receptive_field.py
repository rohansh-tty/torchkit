jump_in = 1
jump_out = 1

def conv_output(in_ch_size=28, padding=0, kernel_size=3, stride=1):
    global jump_out, jump_in
    jump_out = jump_in*stride
    out_ch_size = int((in_ch_size + (2*padding) - kernel_size)/stride) + 1
    return f'OutputCh: {out_ch_size} JumpOut:{jump_out}'

def rf_calc(in_rf=1, kernel_size=3, jump_in=1):
    return in_rf + (kernel_size-1)*jump_in

print(conv_output(in_ch_size=32, stride=2))
print(conv_output(in_ch_size=30, stride=1))
