from utils import calculate_distance


class SpeedTracker:

    def __init__(self):

        self.previous_positions = {}

        self.previous_times = {}

        self.current_speeds = {}

        self.max_speeds = {}



    def update(self, frame_objects):

        """
        Calculate player speed frame by frame
        """

        for obj in frame_objects:


            # Only players
            if obj["class"] != "player":

                continue



            player_id = obj["player_id"]



            current_position = (

                obj["x_pitch"],

                obj["y_pitch"]

            )


            current_time = obj["timestamp"]




            # First detection

            if player_id not in self.previous_positions:


                self.previous_positions[player_id] = current_position

                self.previous_times[player_id] = current_time

                self.current_speeds[player_id] = 0

                self.max_speeds[player_id] = 0


                continue




            previous_position = self.previous_positions[player_id]


            previous_time = self.previous_times[player_id]



            distance = calculate_distance(

                previous_position[0],

                previous_position[1],

                current_position[0],

                current_position[1]

            )



            time_difference = (

                current_time - previous_time

            )



            if time_difference > 0:


                speed = distance / time_difference


                speed = round(speed,2)



                self.current_speeds[player_id] = speed



                if speed > self.max_speeds[player_id]:

                    self.max_speeds[player_id] = speed




            self.previous_positions[player_id] = current_position

            self.previous_times[player_id] = current_time





    def get_current_speeds(self):

        return self.current_speeds





    def get_top_speed_player(self):


        if not self.max_speeds:


            return {

                "player_id": None,

                "speed": 0

            }



        player = max(

            self.max_speeds,

            key=self.max_speeds.get

        )



        return {


            "player_id": player,


            "speed": round(

                self.max_speeds[player],

                2

            )


        }