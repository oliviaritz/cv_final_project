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

for i in range(81):
	# shift to be centered in box
	topleftx = boxes[i][0][0]
	toprightx = boxes[i][2][0]
	difx = (toprightx - topleftx) / 4	# divide by 4 instead of 2 because of how digit is written
	newx = topleftx + difx

	toplefty = boxes[i][0][1]
	botlefty = boxes[i][1][1] 
	dify = (botlefty - toplefty) / 4
	newy = toplefty + dify

	# draw new value onto image
	# TODO: use correct sudoku puzzle output instead of 1
	draw.text((newx, newy),"1",(0,0,0),font=font)

img.save('test-out.jpg')
