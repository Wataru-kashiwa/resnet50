a = 0
b=0
c=3
d=3
ans = 0
for i in range(10**9):
  i0 = -i
  for j in range(10**9):
    j0 = -j
    e = (a-i)**2 + (b-j)**2
    e0 = (a-i0)**2 + (b-j0)**2
    e1 = (a-i)**2 + (b-j0)**2
    e2 = (a-i0)**2 + (b-j)**2
    
    f = (c-i)**2 + (d-j)**2
    f0 = (c-i0)** + (d-j0)**2
    f1 = (c-i)**2 + (d-j0)**2
    f2 = (c-i0)**2 + (d-j)**2
    
    if e==5 and f==5:
      print("Yes")
      ans += 1
      break
    if e0==5 and f0==5:
      print("Yes")
      ans += 1
      break
    if e1==5 and f1==5:
      print("Yes")
      ans += 1
      break
    if e2==5 and f2==5:
      print("Yes")
      ans += 1
      break

if ans == 0:
  print("No")