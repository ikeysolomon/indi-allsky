with open(r'DENOISE PR TEST ENVIRONMENT/test_pr_real_image.py','r',encoding='utf-8') as f:
    lines=f.readlines()
for i in range(320, 341):
    ln=lines[i-1]
    print(i, repr(ln[:40]))
