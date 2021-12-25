import arcade
import arcade.gui
import math


class Player(arcade.Sprite):
    def __init__(self, image, scale):
        super().__init__(image, scale)

        # Create a variable to hold our speed. 'angle' is created by the parent
        self.speed = 0

    def update(self):
        # Convert angle in degrees to radians.
        angle_rad = math.radians(self.angle)

        # Rotate
        self.angle += self.change_angle

        if self.angle >= 360:
            self.angle = self.angle - 360

        # Use math to find our change based on our speed and angle
        self.center_x += -self.speed * math.sin(angle_rad)
        self.center_y += self.speed * math.cos(angle_rad)


class Game(arcade.Window):
    def __init__(self):
        super().__init__(1000, 1000, title="Sieci neurotyczne")
        arcade.set_background_color(arcade.color.LIGHT_GRAY)

        self.velocity = 1
        self.direction = 1

        self.scene = None
        self.player = None
        self.walls = None

        self.playerDots = []
        self.numberOfDots = 8

        self.i = 0

        self.text = ''

        self.ui_manager = arcade.gui.UIManager(self)

        self.camera = None

    def center_camera_to_player(self):
        screen_center_x = self.player.center_x - (self.camera.viewport_width / 2)
        screen_center_y = self.player.center_y - (
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

        arcade.draw_text(f'Direction: {self.player.angle}',
                         start_x=0, start_y=self.height - 55,
                         width=self.width,
                         font_size=24,
                         align="center",
                         color=arcade.color.BLACK)

        self.player.draw_hit_box(color=arcade.color.BLACK)

        for i in range(self.numberOfDots):
            closest = arcade.get_closest_sprite(self.playerDots[i], self.walls)
            arcade.draw_line(self.playerDots[i].center_x, self.playerDots[i].center_y,
                             closest[0].center_x, closest[0].center_y, arcade.color.BLACK, 3)

        # closest2 = arcade.get_closest_sprite(self.player[2], self.walls)
        # closest3 = arcade.get_closest_sprite(self.player[3], self.walls)
        #
        # arcade.draw_text(f'Distances: {closest1[1], closest2[1], closest3[1], }',
        #                  start_x=0, start_y=self.height - 85,
        #                  width=self.width,
        #                  font_size=14,
        #                  align="center",
        #                  color=arcade.color.BLACK)
        #
        # arcade.draw_line(self.player[2].center_x, self.player[2].center_y, closest2[0].center_x, closest2[0].center_y, arcade.color.BLACK, 3)
        # arcade.draw_line(self.player[3].center_x, self.player[3].center_y, closest3[0].center_x, closest3[0].center_y, arcade.color.BLACK, 3)

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
        self.player = Player('imgs/ship.png', 1)

        # Adding coordinates for the center of the sprite
        self.player.center_x = 300
        self.player.center_y = 300

        self.scene.add_sprite('Player', self.player)

        for i in range(self.numberOfDots):
            self.playerDots.append(arcade.SpriteCircle(3, color=arcade.color.ALABAMA_CRIMSON))
            self.playerDots[i].center_x = 0
            self.playerDots[i].center_y = 0
            self.scene.add_sprite(f'Dot{i}', self.playerDots[i])

        for i in range(20):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 0 + (64 * i)
            wall.center_y = 500
            self.walls.append(wall)

        for i in range(5):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 180
            wall.center_y = 180 + (64 * i)
            self.walls.append(wall)

        for i in range(5):
            wall = arcade.Sprite("imgs/stone.png", 0.5)
            wall.center_x = 420
            wall.center_y = 180 + (64 * i)
            self.walls.append(wall)

        self.scene.add_sprite_list('Walls', False, self.walls)

    def on_update(self, delta_time):

        points = self.player.get_adjusted_hit_box()

        for i in range(self.numberOfDots):
            self.playerDots[i].center_x = points[i][0]
            self.playerDots[i].center_y = points[i][1]

        self.player.update()
        # if self.i == 0:
        #     self.player.change_angle = 270
        # else:
        self.player.change_angle = 0
        if self.i % 200:
            self.player.change_angle = -2

        self.player.speed = 0.5

        collision = arcade.check_for_collision_with_list(self.player, self.walls)

        if collision:
            self.player.speed = 0
            self.text = 'DEAD'

        self.i = self.i + 1
        self.center_camera_to_player()


game = Game()
game.setup()
arcade.run()
