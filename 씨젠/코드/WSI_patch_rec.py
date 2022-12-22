import slide_rec, patch_rec

def main ():
    slides = slide_rec.main()
    patch_rec.rec_patch(slides)
    print("DONE!")
    return

if __name__ == '__main__':
    main()