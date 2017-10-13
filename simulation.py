import random
import sys
from collections import deque
from heapq import heappop, heappush
from time import sleep

import numpy as np
import pygame
from numba import jit
from pygame import gfxdraw
from pygame.locals import *

WINDOW_SIZE = (640, 400)
BACKGROUND_COLOR = (0, 0, 0)
BLOCK_SIZE = 10
BLOCKS_X, BLOCKS_Y = WINDOW_SIZE[0] // BLOCK_SIZE, WINDOW_SIZE[1] // BLOCK_SIZE
INIT_SNAKE_LENGTH = 3
food: tuple = None
snake = deque()
points = 0
movement = K_LEFT
visited = np.zeros((BLOCKS_X, BLOCKS_Y), dtype=bool)
added_to_heap = np.zeros((BLOCKS_X, BLOCKS_Y), dtype=bool)
previous = np.zeros((BLOCKS_X, BLOCKS_Y), dtype=np.dtype('int,int'))


class Segment(object):
    _pos = (-1, -1)

    def __init__(self, x: int = None, y: int = None, pos: tuple = None):
        if pos:
            self.x, self.y = pos
        elif x is not None and y is not None:
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


def mark_snake_on_board():
    for segment in snake:
        visited[segment.x, segment.y] = True


def random_pos():
    return random.randint(0, BLOCKS_X - 1), random.randint(0, BLOCKS_Y - 1)


def init_snake():
    global snake
    snake = deque()
    x, y = random_pos()
    for i in range(INIT_SNAKE_LENGTH):
        snake.append(Segment(x, y))


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


def draw_visited(screen):
    for (x, y), vis in np.ndenumerate(visited):
        if vis:
            pygame.gfxdraw.filled_circle(screen, x * BLOCK_SIZE + BLOCK_SIZE // 2, y * BLOCK_SIZE + BLOCK_SIZE // 2,
                                         BLOCK_SIZE // 4, (70, 105, 220))


def draw_scene(screen):
    screen.fill(BACKGROUND_COLOR)
    draw_visited(screen)
    draw_snake(screen)
    draw_food(screen)
    draw_text(screen, "Points: " + str(points), (0, 0))


def move_snake():
    tail = snake.pop()
    x = {
        K_LEFT: -1,
        K_RIGHT: 1,
    }.get(movement, 0)
    y = {
        K_UP: -1,
        K_DOWN: 1,
    }.get(movement, 0)
    tail.pos = (snake[0].x + x, snake[0].y + y)
    if tail.x < 0:
        tail.x = BLOCKS_X - 1
    elif tail.x >= BLOCKS_X:
        tail.x = 0
    elif tail.y < 0:
        tail.y = BLOCKS_Y - 1
    elif tail.y >= BLOCKS_Y:
        tail.y = 0
    snake.appendleft(tail)


def init_food():
    global food
    food = empty_space()


@jit
def neighbours(point: tuple) -> tuple:
    directions = [([0, -1], K_UP), ([0, 1], K_DOWN), ([-1, 0], K_LEFT), ([1, 0], K_RIGHT)]
    np.random.shuffle(directions)
    point = np.asarray(point)
    for vec, key in directions:
        direction = point + vec
        if direction[0] >= BLOCKS_X:
            direction[0] = 0
        elif direction[0] < 0:
            direction[0] = BLOCKS_X - 1
        elif direction[1] >= BLOCKS_Y:
            direction[1] = 0
        elif direction[1] < 0:
            direction[1] = BLOCKS_Y - 1
        yield tuple(direction), key


@jit
def points_2d_dist_torus(a: tuple, b: tuple):
    mn = min(a[0], b[0])
    mx = max(a[0], b[0])
    x_dist = min((mx - mn) ** 2, (mn + BLOCKS_X - mx) ** 2)

    mn = min(a[1], b[1])
    mx = max(a[1], b[1])
    y_dist = min((mx - mn) ** 2, (mn + BLOCKS_Y - mx) ** 2)
    return x_dist + y_dist


def points_2d_dist_euclidean(a: tuple, b: tuple):
    return np.linalg.norm(np.asarray(a) - np.asarray(b))  # We can pass edge so geometry is torus not plane


@jit
def points_2d_dist_manhattan(a: tuple, b: tuple):  # Snake should avoid zig-zag for A*
    mn = min(a[0], b[0])
    mx = max(a[0], b[0])
    x_dist = min((mx - mn), (mn + BLOCKS_X - mx))

    mn = min(a[1], b[1])
    mx = max(a[1], b[1])
    y_dist = min((mx - mn), (mn + BLOCKS_Y - mx))
    return x_dist + y_dist


def any_possible_move():
    visited.fill(False)
    mark_snake_on_board()
    head_pos = snake[0].pos
    for neighbour, direction in neighbours(head_pos):
        if not visited[neighbour]:
            return direction


def find_path(screen):
    global previous, visited
    visited.fill(False)
    added_to_heap.fill(False)
    mark_snake_on_board()
    head = snake[0]
    visited[head.pos] = False
    previous[head.pos] = (-1, -1)
    heap = [(0, head.pos)]
    closest_pos = head.pos
    while closest_pos != food:
        closest_dist, closest_pos = heappop(heap)
        for neighbour, direction in neighbours(closest_pos):
            if not visited[neighbour] and not added_to_heap[neighbour]:
                heappush(heap, (points_2d_dist_manhattan(food, neighbour), neighbour))
                added_to_heap[neighbour] = True
                previous[neighbour] = closest_pos
        visited[closest_pos] = True
        if len(heap) == 0:  # No way to food
            return None
        elif len(heap) == 1:  # Move random if hit bottleneck
            return any_possible_move()

    node = food
    while tuple(previous[node]) != head.pos:
        node = tuple(previous[node])
    for neighbour, direction in neighbours(node):
        if neighbour == head.pos:
            return {K_UP: K_DOWN,
                    K_DOWN: K_UP,
                    K_LEFT: K_RIGHT,
                    K_RIGHT: K_LEFT}[direction]
    return None


def empty_space():
    global visited
    visited.fill(False)
    mark_snake_on_board()
    x, y = np.where(visited == False)
    i = np.random.randint(len(x))
    return x[i], y[i]


def check_food():
    global food
    if food == snake[0].pos:
        global points
        points += 1
        snake.appendleft(Segment(pos=food))
        food = empty_space()


def init_game():
    init_snake()
    init_food()
    global points
    points = 0


def draw_text(screen, text, pos):
    font = pygame.font.SysFont("monospace", 15)
    label = font.render(text, 1, (255, 255, 255))
    screen.blit(label, pos)


def draw_game_over(screen):
    font = pygame.font.SysFont("monospace", 15)
    label = font.render("Game over", 1, (255, 255, 255))
    screen.blit(label, (WINDOW_SIZE[0] // 2 - label.get_width() // 2, WINDOW_SIZE[1] // 2 - label.get_height() // 2))
    label = font.render("Press Enter", 1, (255, 255, 255))
    screen.blit(label, (WINDOW_SIZE[0] // 2 - label.get_width() // 2, WINDOW_SIZE[1] // 2 + label.get_height() // 2))


def main():
    init_game()
    pygame.init()
    pygame.font.init()
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
                elif event.key == K_RETURN or event.key == K_SPACE:
                    init_game()

    while True:
        input(pygame.event.get())
        global movement
        movement = find_path(screen)
        if movement is None:
            draw_scene(screen)
            draw_game_over(screen)
            pygame.display.update()
            sleep(0.5)
        else:
            move_snake()
            check_food()
            draw_scene(screen)
            pygame.display.update()
            pygame.time.delay(80)


if __name__ == '__main__':
    main()
