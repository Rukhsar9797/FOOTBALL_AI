import numpy as np
import cv2


class LiveHeatmap:


    def __init__(self):

        self.ball_points = []

        self.red_points = []

        self.white_points = []



        self.width = 1920

        self.height = 1080




    def update(self, objects):


        for obj in objects:


            x = int(obj.get("x",0))

            y = int(obj.get("y",0))


            if obj["class"] == "ball":


                self.ball_points.append(

                    (x,y)

                )



            elif obj["class"] == "player":


                if obj.get("team") == "Red Team":


                    self.red_points.append(

                        (x,y)

                    )


                elif obj.get("team") == "White Team":


                    self.white_points.append(

                        (x,y)

                    )





    def create_heatmap(self, points):


        heat = np.zeros(

            (self.height,self.width),

            dtype=np.float32

        )



        for x,y in points:


            if 0 <= x < self.width and 0 <= y < self.height:


                cv2.circle(

                    heat,

                    (x,y),

                    50,

                    255,

                    -1

                )




        # blur the points

        heat = cv2.GaussianBlur(

            heat,

            (101,101),

            0

        )



        # normalize

        heat = cv2.normalize(

            heat,

            None,

            0,

            255,

            cv2.NORM_MINMAX

        )



        heat = heat.astype(

            np.uint8

        )



        # convert to visible colors

        heat = cv2.applyColorMap(

            heat,

            cv2.COLORMAP_JET

        )


        return heat





    def get_ball_heatmap(self):


        return self.create_heatmap(

            self.ball_points

        )





    def get_red_heatmap(self):


        return self.create_heatmap(

            self.red_points

        )





    def get_white_heatmap(self):


        return self.create_heatmap(

            self.white_points

        )