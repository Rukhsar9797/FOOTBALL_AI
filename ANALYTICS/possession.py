from utils import calculate_distance



class PossessionTracker:


    def __init__(self):


        # Player possession count

        self.player_possession = {}



        # Team possession count

        self.team_possession = {

            "Red Team": 0,

            "White Team": 0

        }



        # Current player holding ball

        self.current_possessor = None




    def update(self, frame_objects):


        ball = None

        players = []



        # Separate ball and players

        for obj in frame_objects:


            if obj["class"] == "ball":

                ball = obj



            elif obj["class"] == "player":

                players.append(obj)




        if ball is None or len(players) == 0:

            return





        nearest_player = None

        minimum_distance = float("inf")



        # Find closest player to ball

        for player in players:


            distance = calculate_distance(

                ball["x_pitch"],

                ball["y_pitch"],

                player["x_pitch"],

                player["y_pitch"]

            )



            if distance < minimum_distance:


                minimum_distance = distance

                nearest_player = player




        # Possession threshold

        # Player must be within 3 metres

        if nearest_player and minimum_distance <= 3:



            player_id = nearest_player["player_id"]



            self.current_possessor = player_id




            # Player count

            if player_id not in self.player_possession:

                self.player_possession[player_id] = 0



            self.player_possession[player_id] += 1




            # Team count

            team = nearest_player.get("team")



            if team in self.team_possession:


                self.team_possession[team] += 1






    def get_current_possessor(self):

        return self.current_possessor





    def get_player_rankings(self):


        return sorted(

            self.player_possession.items(),

            key=lambda x:x[1],

            reverse=True

        )





    def get_team_percentage(self):


        total = sum(

            self.team_possession.values()

        )



        if total == 0:


            return {

                "Red Team":0,

                "White Team":0

            }





        return {


            team: round(

                (count / total) * 100,

                2

            )


            for team,count in self.team_possession.items()


        }





# ==================================
# TEST POSSESSION MODULE
# ==================================

if __name__ == "__main__":


    from utils import load_json, group_by_frame



    data = load_json(

        "input/detections.json"

    )



    frames = group_by_frame(data)



    tracker = PossessionTracker()



    for frame_number in sorted(frames):


        tracker.update(

            frames[frame_number]

        )




    print("\n==============================")

    print("CURRENT BALL POSSESSOR")

    print("==============================")


    print(

        tracker.get_current_possessor()

    )




    print("\n==============================")

    print("TEAM POSSESSION %")

    print("==============================")


    print(

        tracker.get_team_percentage()

    )




    print("\n==============================")

    print("PLAYER POSSESSION RANKING")

    print("==============================")


    for player,count in tracker.get_player_rankings():


        print(

            f"Player {player}: {count} frames"

        )