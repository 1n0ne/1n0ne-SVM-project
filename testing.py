import pygame
import joblib
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.transform import resize
# loading pre trained model
model=joblib.load('svmRmodel')

#predict function
def predict_digit(img):
    img = resize(imread((img)), (8,8))
    img = rescale_intensity(img, out_range=(0, 16))
    x_test = [sum(pixel)/3.0 for row in img for pixel in row]
    return (model.predict([x_test]))


# pre defined colors, pen radius and font color
black = [255, 255, 255]
white = [0, 0, 0]
red = [255, 0, 0]
green = [0, 255, 0]
draw_on = False
last_pos = (0, 0)
color = (255, 128, 0)
radius = 7
font_size = 500

#image size
width = 256
height = 256

# initializing screen
screen = pygame.display.set_mode((width, height))
screen.fill(white)
pygame.font.init()



def roundline(srf, color, start, end, radius=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(srf, color, (x, y), radius)


def draw_partition_line():
    pygame.draw.line(screen, black, [width, 0], [width,height ], 8)

try:
    while True:
        # get all events
        e = pygame.event.wait()
        draw_partition_line()

        # clear screen after right click
        if(e.type == pygame.MOUSEBUTTONDOWN and e.button == 3):
            screen.fill(white)

        # quit
        if e.type == pygame.QUIT:
            raise StopIteration

        # start drawing after left click
        if(e.type == pygame.MOUSEBUTTONDOWN and e.button != 3):
            color = black
            pygame.draw.circle(screen, color, e.pos, radius)
            draw_on = True

        # stop drawing after releasing left click
        if e.type == pygame.MOUSEBUTTONUP and e.button != 3:
            draw_on = False
            fname = "out.png"
            img = screen
            pygame.image.save(img, fname)
            print("The predicted digit is {}".format(predict_digit("out.png")))

            

        # start drawing line on screen if draw is true
        if e.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(screen, color, e.pos, radius)
                roundline(screen, color, e.pos, last_pos, radius)
            last_pos = e.pos

        pygame.display.flip()

except StopIteration:
    pygame.quit()