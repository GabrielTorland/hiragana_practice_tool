import math

def get_hiragana_characters():
    return [
        HiraganaA()
    ]

class HiraganaCharacter:
    def __init__(self, character, romanji, number_of_strokes):
        self.character = character
        self.romanji = romanji
        self.nr_of_strokes = number_of_strokes
    
    def validate_stroke(self,stroke_number, stroke_points):
        raise NotImplementedError("Subclass must implement abstract method")

    def __str__(self):
        return f"{self.character} - {self.romanji}" 


class HiraganaA(HiraganaCharacter):
    def __init__(self):
        super().__init__("あ", "a", 3)
    
    def validate_stroke(self, stroke_number, stroke_points):
        match stroke_number:
            case 1:
                x_0, y_0 = min(stroke_points, key=lambda point: point[1])
                x_1, y_1 = max(stroke_points, key=lambda point: point[1])

                # Calculate the angle of the stroke
                angle = math.degrees(math.atan2(y_1 - y_0, x_1 - x_0))

                # Check if the angle is within the range of 45 degrees
                is_valid_angle = -45 <= angle <= 45

                return is_valid_angle, "The first stroke should be a horizontal line"

            case 2:
                pass

            case 3:
                pass
        
        return False, "Invalid stroke number"
