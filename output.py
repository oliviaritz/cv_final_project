from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pickle 

img = Image.open("test.jpg")
draw = ImageDraw.Draw(img)

# font = ImageFont.truetype(<font-file>, <font-size>)
font = ImageFont.truetype("arial.ttf", 36)

# boxes is list of corner coordinates for all 81 points
with open('boxes.txt', 'rb') as fp:
        boxes = pickle.load(fp)
		
# fake sudoku results:
result = [0, 2, 5, 4, 0, 0, 0, 3, 9,		# row 1 (top)
		  8, 0, 1, 3, 9, 6, 2, 0, 0,		# row 2
		  3, 0, 0, 7, 5, 2, 0, 4, 1, 		# row 3
		  6, 1, 8, 0, 0, 0, 0, 5, 7,		# row 4
		  0, 5, 0, 1, 6, 7, 4, 8, 0,		# row 5
		  0, 0, 4, 5, 0, 0, 9, 0, 6,		# row 6
		  5, 3, 7, 2, 0, 9, 0, 6, 8,		# row 7
		  4, 8, 0, 0, 7, 1, 5, 9, 0,		# row 8
		  1, 0, 0, 0, 3, 5, 7, 2, 4]		# row 9 (bot)
		  
		  

#print(boxes[0][0])
#draw.text(boxes[0][0], "1" ,(150,0,0),font=font)
#print(boxes[0][1])
#draw.text(boxes[0][1], "2" ,(150,0,0),font=font)
#print(boxes[0][2])
#draw.text(boxes[0][2], "3" ,(150,0,0),font=font)
#print(boxes[0][3])
#draw.text(boxes[0][3], "4" ,(150,0,0),font=font)


for i in range(81):
	# shift to be centered in box
	topleftx = boxes[i][0][0]
	toprightx = boxes[i][1][0]
	difx = (toprightx - topleftx) / 4	# divide by 4 instead of 2 because of how digit is written
	newx = topleftx + difx 

	toplefty = boxes[i][0][1]
	botlefty = boxes[i][2][1] 
	dify = (botlefty - toplefty) / 8
	newy = toplefty + dify

	# draw new value onto image
	# TODO: use correct sudoku puzzle output instead of 1
	if result[i] != 0:
		draw.text((newx, newy),str(result[i]),(150,0,0),font=font)

img.save('test-out.jpg')
