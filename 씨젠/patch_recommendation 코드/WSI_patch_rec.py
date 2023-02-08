import slide_rec, patch_rec

def main ():
    slides = slide_rec.main() # recommendation slide 정보 반환 
    patch_rec.rec_patch(slides) # 신뢰도가 낮은 patch 및 주변의 patch들을 선별한다. 
    
    # Q. 호출된 추천 list는 어디에 포함이 되는 것인가? 어디에도 결과가 안 남는 것 같은데? 
    print("DONE!") 
    return

if __name__ == '__main__':
    main()