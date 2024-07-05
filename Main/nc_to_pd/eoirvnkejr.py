def main(px, py, pz, alpha, phi=None):
    result = [px, py, pz, alpha, phi]
    return(result)


test = main(px=3, py=5,pz=3, alpha=45, phi=45)
print(test) 