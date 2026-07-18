from utils import calculate_distance


class DistanceTracker:

    def __init__(self):

        # Previous position of each player
        self.previous_positions = {}

        # Total distance covered by each player
        self.player_distances = {}



    def update(self, frame_objects):

        """
        Process one frame of detections
        """

        for obj in frame_objects:


            # Ignore ball
            if obj["class"] != "player":
                continue


            player_id = obj["player_id"]


            current_position = (

                obj["x_pitch"],

                obj["y_pitch"]

            )


            # First time seeing player

            if player_id not in self.player_distances:

                self.player_distances[player_id] = 0



            # Calculate movement

            if player_id in self.previous_positions:


                previous_position = self.previous_positions[player_id]


                distance = calculate_distance(

                    previous_position[0],

                    previous_position[1],

                    current_position[0],

                    current_position[1]

                )


                self.player_distances[player_id] += distance



            # Update position

            self.previous_positions[player_id] = current_position




    def get_rankings(self):

        """
        Returns all players sorted by distance
        """

        return sorted(

            self.player_distances.items(),

            key=lambda x: x[1],

            reverse=True

        )




    def get_top_player(self):

        """
        Returns highest distance player
        """

        if not self.player_distances:

            return None



        player = max(

            self.player_distances,

            key=self.player_distances.get

        )


        return {

            "player_id": player,

            "distance": round(

                self.player_distances[player],

                2

            )

        }





# ==============================
# TESTING DISTANCE MODULE
# ==============================

if __name__ == "__main__":


    from utils import load_json, group_by_frame



    data = load_json(

        "input/detections.json"

    )


    frames = group_by_frame(data)



    tracker = DistanceTracker()



    for frame_number in sorted(frames):


        tracker.update(

            frames[frame_number]

        )



    print("\n==============================")

    print("DISTANCE RANKING")

    print("==============================")



    for player, distance in tracker.get_rankings():


        print(

            f"Player {player}: {distance:.2f} meters"

        )



    print("\n==============================")

    print("TOP DISTANCE PLAYER")

    print("==============================")


    print(

        tracker.get_top_player()

    )