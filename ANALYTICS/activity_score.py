class ActivityTracker:


    def __init__(self):

        # Store player activity scores

        self.activity_scores = {}





    def update(self,
               distance_tracker,
               speed_tracker,
               possession_tracker):


        distances = (

            distance_tracker.player_distances

        )


        speeds = (

            speed_tracker.max_speeds

        )


        possessions = (

            possession_tracker.player_possession

        )



        # Collect all players

        players = set()



        players.update(

            distances.keys()

        )


        players.update(

            speeds.keys()

        )


        players.update(

            possessions.keys()

        )





        for player_id in players:



            distance = distances.get(

                player_id,

                0

            )



            speed = speeds.get(

                player_id,

                0

            )



            possession = possessions.get(

                player_id,

                0

            )




            score = (

                (0.4 * distance)

                +

                (0.4 * speed)

                +

                (0.2 * possession)

            )



            self.activity_scores[player_id] = round(

                score,

                2

            )







    def get_ranking(self):


        return sorted(

            self.activity_scores.items(),

            key=lambda x:x[1],

            reverse=True

        )






    def get_most_active_player(self):


        if not self.activity_scores:

            return None



        player = max(

            self.activity_scores,

            key=self.activity_scores.get

        )



        return {


            "player_id": player,


            "activity_score": self.activity_scores[player]


        }








# ==================================
# TEST ACTIVITY MODULE
# ==================================

if __name__ == "__main__":


    from utils import load_json, group_by_frame

    from distance import DistanceTracker

    from speed import SpeedTracker

    from possession import PossessionTracker




    data = load_json(

        "input/detections.json"

    )



    frames = group_by_frame(data)




    distance_tracker = DistanceTracker()

    speed_tracker = SpeedTracker()

    possession_tracker = PossessionTracker()



    for frame_number in sorted(frames):


        objects = frames[frame_number]


        distance_tracker.update(objects)

        speed_tracker.update(objects)

        possession_tracker.update(objects)





    activity_tracker = ActivityTracker()



    activity_tracker.update(

        distance_tracker,

        speed_tracker,

        possession_tracker

    )




    print("\n==============================")

    print("ACTIVITY RANKING")

    print("==============================")




    for player,score in activity_tracker.get_ranking():


        print(

            f"Player {player}: {score}"

        )





    print("\n==============================")

    print("MOST ACTIVE PLAYER")

    print("==============================")



    print(

        activity_tracker.get_most_active_player()

    )