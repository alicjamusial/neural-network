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

        # Use math to find our change based on our speed and angle
        self.center_x += -self.speed * math.sin(angle_rad)
        self.center_y += self.speed * math.cos(angle_rad)


class Game(arcade.Window):
    def __init__(self):
        super().__init__(600, 600, title="Sieci neurotyczne")
        arcade.set_background_color(arcade.color.LIGHT_GRAY)

        self.velocity = 1
        self.direction = 1

        self.scene = None
        self.player = None

        self.ui_manager = arcade.gui.UIManager(self)

    # Creating on_draw() function to draw on the screen
    def on_draw(self):
        arcade.start_render()

        self.scene.draw()

        self.ui_manager.draw()
        arcade.draw_text(f'Direction: {self.player_sprite.angle}',
                         start_x=0, start_y=self.height - 55,
                         width=self.width,
                         font_size=24,
                         align="center",
                         color=arcade.color.BLACK)

    def setup(self):

        self.scene = arcade.Scene()

        self.player_sprite = Player('imgs/p.png', 1)

        # Adding coordinates for the center of the sprite
        self.player_sprite.center_x = 300
        self.player_sprite.center_y = 300

        # Adding sprites in scene
        self.scene.add_sprite('Player', self.player_sprite)

    def on_update(self, delta_time):

        self.player_sprite.update()

        # self.player_sprite.change_angle = 2
        self.player_sprite.speed = 1

        # self.player_sprite.center_x += self.velocity * delta_time
        # self.player_sprite.angle += self.direction * 360

        # # Checking if sprites are colliding or not
        # colliding = arcade.check_for_collision(
        #     self.player_sprite, self.player_sprite2)
        #
        # # If sprites are colliding then changing direction
        # if colliding:
        #     self.vel_x1 *= -1
        #     self.vel_x2 *= -1
        #
        # # Changing the direction if sprites crosses the screen boundary
        # if self.player_sprite.center_x > 600 or self.player_sprite.center_x < 0:
        #     self.vel_x1 *= -1
        #
        # if self.player_sprite2.center_x > 600 or self.player_sprite2.center_x < 0:
        #     self.vel_x2 *= -1


game = Game()
game.setup()
arcade.run()