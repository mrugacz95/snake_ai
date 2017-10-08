import random
import sys
from collections import deque

import pygame
from pygame.locals import *

WINDOW_SIZE = (800, 480)
BACKGROUND_COLOR = (0, 0, 0)
SEGMENT_SIZE = 10
food: tuple = None


class Segment(object):
    _pos = None

    def __init__(self, pos: tuple):
        self.x, self.y = pos

    def __init__(self, x: int, y: int):
        self.x, self.y = x, y

    @property
    def pos(self) -> tuple:
        return self.x, self.y

    @pos.setter
    def pos(self, pos):
        x, y = pos
        self.x = x
        self.y = y


snake = deque()

direction = K_LEFT


def init_snake():
    x, y = (random.randint(0, WINDOW_SIZE[0] // SEGMENT_SIZE),
            random.randint(0, WINDOW_SIZE[1] // SEGMENT_SIZE))
    snake.append(Segment(x, y))
    snake.append(Segment(x + 1, y))
    snake.append(Segment(x + 2, y))


def draw_segment(screen: pygame.Surface, segment: Segment):
    x, y = segment.pos
    pygame.draw.rect(screen, (50, 180, 50), (x * SEGMENT_SIZE, y * SEGMENT_SIZE, SEGMENT_SIZE, SEGMENT_SIZE))


def draw_snake(screen: pygame.Surface):
    for segment in snake:
        draw_segment(screen, segment)


def draw_food(screen: pygame.Surface):
    x, y = food
    x, y = x * SEGMENT_SIZE + SEGMENT_SIZE // 2, y * SEGMENT_SIZE + SEGMENT_SIZE // 2
    pygame.draw.circle(screen, (200, 50, 60), (x, y), SEGMENT_SIZE // 2)


def draw_scene(screen):
    screen.fill(BACKGROUND_COLOR)
    draw_snake(screen)
    draw_food(screen)
    pygame.display.update()


def move_snake():
    global direction
    tail = snake.pop()
    x = {
        K_LEFT: -1,
        K_RIGHT: 1,
    }.get(direction, 0)
    y = {
        K_UP: -1,
        K_DOWN: 1,
    }.get(direction, 0)
    tail.pos = (snake[0].x + x, snake[0].y + y)
    if tail.x < 0:
        tail.x = WINDOW_SIZE[0] // SEGMENT_SIZE
    elif tail.x > WINDOW_SIZE[0] // SEGMENT_SIZE:
        tail.x = 0
    elif tail.y < 0:
        tail.y = WINDOW_SIZE[1] // SEGMENT_SIZE
    elif tail.y > WINDOW_SIZE[1] // SEGMENT_SIZE:
        tail.y = 0
    snake.appendleft(tail)


def init_food():
    global food
    food = (3, 4)


def main():
    init_snake()
    init_food()
    pygame.init()
    screen = pygame.display.set_mode((800, 480))

    def input(events):
        for event in events:
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                allowed = {K_UP, K_DOWN, K_LEFT, K_RIGHT}
                if event.key in allowed:
                    global direction
                    direction = event.key
                elif event.key == K_ESCAPE:
                    sys.exit(0)

    while True:
        input(pygame.event.get())
        move_snake()
        draw_scene(screen)
        pygame.time.delay(100)


if __name__ == '__main__':
    main()
