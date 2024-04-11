class HiraganaCharacter:
    def __init__(self, character, romanji, number_of_strokes):
        self.character = character
        self.romanji = romanji
        self.nr_of_strokes = number_of_strokes
    
    def __str__(self):
        return f"{self.character} - {self.romanji}" 


    def __str__(self):
        return f"{self.character} - {self.romanji}" 

def get_hiragana_characters():
    characters = [
        HiraganaCharacter("あ", "a", 3),
        HiraganaCharacter("い", "i", 2),
        HiraganaCharacter("う", "u", 2),
        HiraganaCharacter("え", "e", 2),
        HiraganaCharacter("お", "o", 3),
        # K row
        HiraganaCharacter("か", "ka", 3),
        HiraganaCharacter("き", "ki", 3),
        HiraganaCharacter("く", "ku", 1),
        HiraganaCharacter("け", "ke", 3),
        HiraganaCharacter("こ", "ko", 2),
        # S row
        HiraganaCharacter("さ", "sa", 3),
        HiraganaCharacter("し", "shi", 1),
        HiraganaCharacter("す", "su", 2),
        HiraganaCharacter("せ", "se", 3),
        HiraganaCharacter("そ", "so", 1),
        # and so on for T, N, H, M, Y, R, W rows, and "n" sound
        HiraganaCharacter("た", "ta", 4),
        HiraganaCharacter("な", "na", 4),
        HiraganaCharacter("は", "ha", 3),
        HiraganaCharacter("ま", "ma", 3),
        HiraganaCharacter("や", "ya", 3),
        HiraganaCharacter("ら", "ra", 2),
        HiraganaCharacter("わ", "wa", 2),
        HiraganaCharacter("ん", "n", 1)
    ]
    return characters