import random
import sys
from collections import deque
from enum import Enum
from heapq import heappop, heappush

import numpy as np
import pygame
from pygame import gfxdraw
from pygame.locals import *

WINDOW_SIZE = (100, 100)
BACKGROUND_COLOR = (0, 0, 0)
BLOCK_SIZE = 10
BLOCKS_H, BLOCKS_V = WINDOW_SIZE[0] // BLOCK_SIZE, WINDOW_SIZE[1] // BLOCK_SIZE
food: tuple = None


class Direction(Enum):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


visited = np.zeros((BLOCKS_H, BLOCKS_V), dtype=bool)
previous = np.zeros((BLOCKS_H, BLOCKS_V), dtype=np.dtype('int,int'))


class Segment(object):
    _pos = None

    def __init__(self, x: int = None, y: int = None, pos: tuple = None):
        if pos:
            self.x, self.y = pos
        elif x and y:
            self.x, self.y = x, y
        else:
            raise RuntimeError("Wrong segment initialization")

    @property
    def pos(self) -> tuple:
        return self.x, self.y

    @pos.setter
    def pos(self, pos):
        x, y = pos
        self.x = x
        self.y = y


snake = deque()

movement = K_LEFT


def mark_snake_on_board():
    for segment in snake:
        visited[segment.x, segment.y] = True


def random_pos():
    return random.randint(0, BLOCKS_H - 1), random.randint(0, BLOCKS_V - 1)


def init_snake():
    x, y = 5, 6
    snake.append(Segment(x, y))
    snake.append(Segment(x + 1, y))
    snake.append(Segment(x + 2, y))


def draw_segment(screen: pygame.Surface, segment: Segment):
    x, y = segment.pos
    pygame.draw.rect(screen, (50, 180, 50), (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))


def draw_snake(screen: pygame.Surface):
    for segment in snake:
        draw_segment(screen, segment)


def draw_food(screen: pygame.Surface):
    x, y = food
    x, y = x * BLOCK_SIZE + BLOCK_SIZE // 2, y * BLOCK_SIZE + BLOCK_SIZE // 2
    pygame.gfxdraw.filled_circle(screen, x, y, BLOCK_SIZE // 2, (200, 50, 60))


def draw_scene(screen):
    screen.fill(BACKGROUND_COLOR)
    draw_snake(screen)
    draw_food(screen)
    pygame.display.update()


def move_snake():
    tail = snake.pop()
    x = {
        K_LEFT: -1,
        K_RIGHT: 1,
        Direction.LEFT: -1,
        Direction.RIGHT: 1,
    }.get(movement, 0)
    y = {
        K_UP: -1,
        K_DOWN: 1,
        Direction.UP: -1,
        Direction.DOWN: 1,
    }.get(movement, 0)
    tail.pos = (snake[0].x + x, snake[0].y + y)
    if tail.x < 0:
        tail.x = BLOCKS_H - 1
    elif tail.x >= BLOCKS_H:
        tail.x = 0
    elif tail.y < 0:
        tail.y = BLOCKS_V - 1
    elif tail.y >= BLOCKS_V:
        tail.y = 0
    snake.appendleft(tail)


def init_food():
    global food
    food = (3, 4)


def neighbours(point: tuple) -> tuple:
    directions = [([0, -1], K_UP), ([0, 1], K_DOWN), ([-1, 0], K_LEFT), ([1, 0], K_RIGHT)]
    point = np.asarray(point)
    for vec, dir in directions:
        direction = point + vec
        if direction[0] >= BLOCKS_H:
            direction[0] = 0
        elif direction[0] < 0:
            direction[0] = BLOCKS_H - 1
        elif direction[1] >= BLOCKS_V:
            direction[1] = 0
        elif direction[1] < 0:
            direction[1] = BLOCKS_V - 1
        yield tuple(direction), dir


def find_path():
    global distance, previous, visited
    visited.fill(False)
    mark_snake_on_board()
    head = snake[0]
    visited[head.pos] = False
    previous[head.pos] = (-1, -1)
    heap = [(0, head.pos)]
    closest_pos = head.pos
    while closest_pos != food and len(heap) != 0:
        closest_dist, closest_pos = heappop(heap)
        for neighbour, direction in neighbours(closest_pos):
            if not visited[neighbour]:
                heappush(heap, (closest_dist + 1, neighbour))
                previous[neighbour] = closest_pos
        visited[closest_pos] = True
    node = food
    while tuple(previous[node]) != head.pos:
        node = tuple(previous[node])
    for neighbour, direction in neighbours(node):
        if neighbour == head.pos:
            return {K_UP: K_DOWN,
                    K_DOWN: K_UP,
                    K_LEFT: K_RIGHT,
                    K_RIGHT: K_LEFT}[direction]
    return Direction.NONE


def check_food():
    global food
    if food == snake[0].pos:
        snake.appendleft(Segment(pos=food))
        global visited
        visited.fill(False)
        mark_snake_on_board()
        x, y = np.where(visited == False)
        i = np.random.randint(len(x))
        random_position = (x[i], y[i])
        food = random_position


def main():
    init_snake()
    init_food()
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)

    def input(events):
        for event in events:
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                allowed = {K_UP, K_DOWN, K_LEFT, K_RIGHT}
                if event.key in allowed:
                    global movement
                    movement = event.key
                elif event.key == K_ESCAPE:
                    sys.exit(0)

    while True:
        input(pygame.event.get())
        global movement
        movement = find_path()
        move_snake()
        check_food()
        draw_scene(screen)
        pygame.time.delay(400)


if __name__ == '__main__':
    main()
