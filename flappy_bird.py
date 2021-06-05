import pygame
import neat
import time
import os
import random

pygame.font.init()
WIN_WIDTH = 500
WIN_HEIGHT = 800

# scale2x: makes the image 2 times bigger

bird_imgs = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]

base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
pip_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
bg = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))  # background img
stat_font = pygame.font.SysFont("comicsans", 50)


class Bird:
    imgs = bird_imgs
    max_rotation = 25  # how much the bird is gonna tilt
    rot_vel = 20
    animation_time = 5  # how long show each bird animation

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0  # its zero to start the bird flat looking
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0  # to keep track of the images of the bird
        self.img = self.imgs[0]  # reference the first bird image in the list (line 12

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1  # it means a tick went down and the bird moved for a sec
        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2  # -10.5 + 1.5 = -9 that means we are moving upwards by 9 pixels

        if d >= 16:  # if it moving down more than 16p its gonna reset it to 16
            d = 16
        if d < 0:  # if it moving upwards move it a little more, why 2? its gonna make it nice jump u can change it
            d -= 2
        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:  # tilt the bird upwards, remember less than 0 means its going up
            if self.tilt < self.max_rotation:
                self.tilt = self.max_rotation  # it sets the rotation of the bird to 25deg
        else:
            if self.tilt > -90:
                self.tilt -= self.rot_vel  # its rotate the bird completely 90 deg

    def draw(self, win):
        self.img_count += 1  # to keep tracking the bird to show the image we need
        if self.img_count < self.animation_time:  # it will show the first image of the bird
            self.img = self.imgs[0]

        elif self.img_count < self.animation_time * 2:
            self.img = self.imgs[1]

        elif self.img_count < self.animation_time * 3:  # if we are less than 15 it will show the last image
            self.img = self.imgs[2]

        elif self.img_count < self.animation_time * 4:  # if we are more than 15 it will show the 2nd image
            self.img = self.imgs[1]

        elif self.img_count == self.animation_time * 4 + 1:
            self.img = self.imgs[0]
            self.img_count = 0

        if self.tilt <= -80:  # animation optimisation
            self.img = self.imgs[1]
            self.img_count = self.animation_time * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)

        # rotates the image around the center GOD knows how
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)  # blit means draw

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    gap = 200
    vel = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0

        # flip the pipe because we have only one pic for the pipe
        self.PIPE_TOP = pygame.transform.flip(pip_img, False, True)

        self.PIPE_BOTTOM = pip_img
        self.passed = False  # if the bird passes the pipe its for the collision
        self.set_height()

    # sets the height of the pipe
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.gap

    def move(self):
        self.x -= self.vel

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):  # comment it later 3rd vod
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.img)
        bottom_mask = pygame.mask.from_surface(self.img)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False


class Base:
    vel = 5
    width = bg.get_width()
    img = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    # moves the base image so it look like infant image
    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel

        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width

        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, win):
        win.blit(self.img, (self.x1, self.y))
        win.blit(self.img, (self.x2, self.y))


# it will creat the window
def draw_window(win, bird, pipes, base, score):
    win.blit(bg, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = stat_font.render("Score :" + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()


def main(genomes, config):
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()  # sets the frame rate
    score = 0
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate(
                (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] >= 0.5:
                bird.jump()

        # bird.move()
        add_pipe = False
        remove = []
        for pipe in pipes:
            for x, bird in enumerate(birds):  # checking every bird is collide with the pipes
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                remove.append(pipe)

            pipe.move()
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append((Pipe(600)))

        for r in remove:
            pipes.remove(r)

        # checks if the bird hits the ground
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()

        draw_window(win, birds, pipes, base, score)






def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)  # population
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
