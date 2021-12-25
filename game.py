from datetime import datetime
from random import random, seed, randrange
from typing import List, Optional
from sklearn import preprocessing
import numpy as np

import arcade
import arcade.gui
import math

from script import NeuronNetwork


class Player(arcade.Sprite):
    def __init__(self, image, scale):
        super().__init__(image, scale)

        # Create a variable to hold our speed. 'angle' is created by the parent
        self.speed = 0

        self.dead = False

        self.playerDots = []
        self.closest: List[Optional[arcade.Sprite]] = []
        self.numberOfDots = 8

        for i in range(self.numberOfDots):
            self.closest.append(None)

        self.network = NeuronNetwork(3, [self.numberOfDots, 16, 2])

    def update(self, walls):
        points = self.get_adjusted_hit_box()

        for i in range(self.numberOfDots):
            self.closest[i] = arcade.get_closest_sprite(self.playerDots[i], walls)

        distances = [i[1] for i in self.closest]
        # normalize the data attributes
        normalized = preprocessing.normalize(np.array(distances).reshape(1,-1))
        self.network.forward_propagate(normalized[0])
        self.speed = (self.network.layers[1].neurons[0].activation + 0.5) * 8
        self.change_angle = self.network.layers[1].neurons[1].activation - 0.5

        for i in range(self.numberOfDots):
            self.playerDots[i].center_x = points[i][0]
            self.playerDots[i].center_y = points[i][1]

        # Convert angle in degrees to radians.
        angle_rad = math.radians(self.angle)

        # Rotate
        self.angle += self.change_angle

        if self.angle >= 360:
            self.angle = self.angle - 360

        # Use math to find our change based on our speed and angle
        self.center_x += -self.speed * math.sin(angle_rad)
        self.center_y += self.speed * math.cos(angle_rad)

    def setup(self, scene):
        for i in range(self.numberOfDots):
            self.playerDots.append(arcade.SpriteCircle(3, color=arcade.color.ALABAMA_CRIMSON))
            self.playerDots[i].center_x = 0
            self.playerDots[i].center_y = 0
            scene.add_sprite(f'Dot{i}', self.playerDots[i])

        self.center_x = 256
        self.center_y = 256

    def draw_all_points(self):
        self.draw_hit_box(color=arcade.color.BLACK)

        for i in range(self.numberOfDots):
            arcade.draw_line(self.playerDots[i].center_x, self.playerDots[i].center_y,
                             self.closest[i][0].center_x, self.closest[i][0].center_y, arcade.color.BLACK, 3)


class Game(arcade.Window):
    def __init__(self):
        super().__init__(1000, 1000, title="Sieci neurotyczne")
        arcade.set_background_color(arcade.color.LIGHT_GRAY)

        self.velocity = 1
        self.direction = 1

        self.scene = None
        self.numberOfPlayers = 12
        self.players: List[Player] = []
        self.walls = None

        self.dead_players = 0

        self.iteration = 0

        self.text = ''

        self.ui_manager = arcade.gui.UIManager(self)

        self.camera = None

        self.chosen = []

    def center_camera_to_player(self):
        most_right_player = self.players[0]
        for i in range(self.numberOfPlayers):
            if self.players[i].center_x > most_right_player.center_x and not self.players[i].dead:
                most_right_player = self.players[i]

        screen_center_x = most_right_player.center_x - (self.camera.viewport_width / 2)
        screen_center_y = most_right_player.center_y - (
                self.camera.viewport_height / 2
        )

        # Don't let camera travel past 0
        if screen_center_x < 0:
            screen_center_x = 0
        if screen_center_y < 0:
            screen_center_y = 0
        player_centered = screen_center_x, screen_center_y

        self.camera.move_to(player_centered)

    # Creating on_draw() function to draw on the screen
    def on_draw(self):
        arcade.start_render()

        self.scene.draw()
        self.camera.use()
        self.ui_manager.draw()

        arcade.draw_text(f'Iteration: {self.iteration}',
                         start_x=0, start_y=self.camera.viewport_height - 55,
                         width=self.camera.viewport_width,
                         font_size=24,
                         align="center",
                         color=arcade.color.BLACK)

        for i in range(self.numberOfPlayers):
            self.players[i].draw_all_points()

        arcade.draw_text(f'{self.text}',
                         start_x=0, start_y=self.height / 2,
                         width=self.width,
                         font_size=44,
                         align="center",
                         color=arcade.color.BLACK)

    def setup(self):
        self.camera = arcade.Camera(self.width, self.height)

        self.scene = arcade.Scene()

        self.walls = arcade.SpriteList()

        for i in range(self.numberOfPlayers):
            self.players.append(Player('imgs/ship.png', 1))
            self.players[i].setup(self.scene)
            self.scene.add_sprite(f'Player{i}', self.players[i])

        for i in range(40):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 0 + (64 * i)
            wall.center_y = 64
            self.walls.append(wall)

        for i in range(40):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 0 + (64 * i)
            wall.center_y = 64 * 14
            self.walls.append(wall)

        for i in range(14):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 32
            wall.center_y = 64 + (64 * i)
            self.walls.append(wall)

        for i in range(10):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 64 * 6
            wall.center_y = 64 + (64 * i)
            self.walls.append(wall)

        for i in range(8):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 64 * 12
            wall.center_y = 64 * 6 + (64 * i)
            self.walls.append(wall)

        for i in range(8):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 64 * 16
            wall.center_y = 64 * 6 + (64 * i)
            self.walls.append(wall)

        for i in range(10):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 64 * 22
            wall.center_y = 64 + (64 * i)
            self.walls.append(wall)

        self.scene.add_sprite_list('Walls', False, self.walls)

        self.end = arcade.SpriteCircle(20, color=arcade.color.ARSENIC)
        self.end.center_x = 2000
        self.end.center_y = 500
        self.scene.add_sprite('End', self.end)

    def on_mouse_press(self, x, y, button, modifiers):
        if self.dead_players == self.numberOfPlayers:
            for i in range(self.numberOfPlayers):
                if self.players[i].collides_with_point((float(x), float(y))):
                    self.players[i].alpha = 255
                    self.chosen.append(self.players[i])

        if len(self.chosen) >= 2:
            self.create_children(self.chosen[0:2])
            self.chosen = []

    def create_children(self, chosen):
        neurons0 = chosen[0].network.layers[0].neurons
        neurons1 = chosen[1].network.layers[0].neurons
        for i in range(self.numberOfPlayers):
            line = randrange(1, 16)
            self.players[i].network.layers[0].neurons[0:line] = neurons0[0:line]
            self.players[i].network.layers[0].neurons[line:16] = neurons1[line:16]

            self.players[i].center_x = 256
            self.players[i].center_y = 256
            self.players[i].dead = False
            self.players[i].alpha = 255
        self.dead_players = 0
        self.iteration += 1

    def on_update(self, delta_time):
        for i in range(self.numberOfPlayers):
            if not self.players[i].dead:
                self.players[i].update(self.walls)
                collision = arcade.check_for_collision_with_list(self.players[i], self.walls)

                if collision:
                    self.players[i].speed = 0
                    self.players[i].change_angle = 0
                    self.players[i].alpha = 20
                    self.players[i].dead = True
                    self.dead_players += 1

        # self.center_camera_to_player()


seed(datetime.now())
game = Game()
game.setup()
arcade.run()
