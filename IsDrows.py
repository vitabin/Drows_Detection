import time

def isDrows(eyescore:int, degreescore:list, undetectframe:int):
    caution_eye, caution_degree = False, False
    # 1 = Drowsinees, 2 = Look forward
    if eyescore >= 50:
        caution_eye = True
        if eyescore >= 100:
            return 1

    if degreescore[0] >= 35 or degreescore[1] >= 25:
        caution_degree = True
        if degreescore[0] >= 450:
            return 2
        elif 450 > degreescore[0] >= 80 or degreescore[1] >= 80:
            return 1

    if caution_eye and caution_degree:
        return 1

    if undetectframe:
        return 2

    