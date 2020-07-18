import pygame
import neat
import time
import os
import random
import math
from pygame.locals import *
pygame.init()
pygame.font.init()
WIN_WIDTH=1000
WIN_HEIGHT=553
image=pygame.image.load(os.path.join("imgs","bird.png"))
bird_img= pygame.transform.scale(image, (80, 80))
pipe_img= pygame.image.load(os.path.join("imgs","cactus.png"))
base_img=pygame.image.load(os.path.join("imgs","base.png"))
bg_img=pygame.image.load(os.path.join("imgs","background.png"))
stat_font= pygame.font.SysFont("comicsans",50)
VEL=10
class Bird:
    imgs=bird_img
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.tick_count=0
        self.img_count = 0
        self.vel=0
        self.height= self.y
        self.img=self.imgs
        self.d=0
        self.theta=0
        self.flag=0
        self.time=0
        self.free=1
        self.jump=0
    def setjump(self):
        self.jump=1
    def jump(self):
        self.vel=-30
        self.tick_count=0
        self.height= self.y
    def move(self):
        self.free=0
        self.time+=.25
        self.theta=(self.time)*math.pi/12
        #print(self.theta)
        if((self.theta<=math.pi)):

            self.y = 473-(173*math.sin(self.theta))
        else:
            self.time=0
            self.jump=0
            self.free=1
    def reset(self):
        self.theta=0
        self.flag=0
        self.t=0

    def draw(self,win):
        self.img_count+=1
        self.img= self.imgs
        win.blit(self.imgs,(self.x,int(self.y)))
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    global VEL

    GAP=200
    vel=VEL
    def __init__(self,x):
        self.x=x
        self.height=0
        self.gap=100
        self.top=0
        self.bottom=0
        self.pipe= pipe_img
        self.passed= False
        self.bottom = self.pipe.get_height()
        self.set_height()

    def set_height(self):

        self.pipe=pygame.transform.scale(pipe_img, (100, 100))

    def move(self):
        global VEL
        self.vel=VEL
        self.x-= self.vel


    def draw(self,win):
        win.blit(self.pipe, (self.x, 553-self.pipe.get_height()))

    def collide(self, bird):
        bird_mask=bird.get_mask()
        pipe_mask= pygame.mask.from_surface(self.pipe)
        offset= (self.x-bird.x, 553-self.pipe.get_height()-round(bird.y))
        b_point= bird_mask.overlap(pipe_mask, offset)
        if(b_point):
            return True
        return False


class Base:
    global VEL
    vel =VEL
    WIDTH = bg_img.get_width()
    IMG = bg_img
    def __init__(self):
        self.y = 0
        self.x1 = 0
        self.x2 = self.WIDTH
    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        global VEL
        self.vel=VEL

        self.x1 -= self.vel
        self.x2 -= self.vel
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    def draw(self, win):
        """
        Draw the floor. This is two images that move together.
        :param win: the pygame surface/window
        :return: None
        """
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))



def draw_window(win,birds,pipes,base,score):
    win.blit(bg_img,(0,0))
    base.draw(win)
    for pipe in pipes:
        pipe.draw(win)
    text=stat_font.render("score: "+str(score),1,(0,0,0))
    win.blit(text,(WIN_WIDTH-10-text.get_width(),10))
    for bird in birds:
        bird.draw(win)
    pygame.display.update()


def main(genomes,config):
    global VEL
    nets=[]
    ge=[]
    birds=[]
    flag=1
    jump=0
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(20,473))
        ge.append(genome)


    base=Base()
    pipes=[Pipe(700)]
    win= pygame.display.set_mode((WIN_WIDTH,WIN_HEIGHT))
    clock = pygame.time.Clock()
    run= True
    score=0
    prev=0
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type== pygame.QUIT:
                run= False
                pygame.quit()
                quit()
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipe.get_width():  # determine whether to use the first or second
                pipe_ind = 1
        else:
            run=False
            break

        print(VEL)
        for x,bird in enumerate(birds):
            #bird.move()
            if bird.free:
                ge[x].fitness +=.1
                output = nets[birds.index(bird)].activate((bird.x,abs(bird.x - pipes[pipe_ind].x)))
                if(output[0]>0.5):

                    bird.setjump()
            if(bird.jump==1):
                bird.move()


            if output[0] > 0.5 and flag==1:
                    jump=1
            else:
                jump=0
            if jump==1:
                flag=bird.move()



        rem =[]

        for pipe in pipes:
            prevscore=score

            for x,bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness-=1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                if pipe.x<bird.x and prev != pipe:
                    prev=pipe
                    score +=1

                    for g in ge:
                        g.fitness+=10

            pipe.move()
            add_pipe=False
            inc=False
            a=0
            if pipe.x + pipe.pipe.get_width()<0:
                rem.append(pipe)

            if not pipe.passed and pipe.x <400:
                pipe.passed=True
                a=1
                add_pipe=True

            if add_pipe:
                pipes.append(Pipe(1100))
            for r in rem:
                pipes.remove(r)
        base.move()
        draw_window(win,birds,pipes,base,score)




def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(main, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'imgs/config-feedforward.txt')
    run(config_path)





