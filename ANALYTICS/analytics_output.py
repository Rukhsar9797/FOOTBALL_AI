import json
import os


def save_analytics(
        possession_tracker,
        distance_tracker,
        speed_tracker,
        activity_tracker,
        momentum_tracker
):

    # Get analytics from trackers

    possession = possession_tracker.get_team_percentage()

    distance_rank = distance_tracker.get_rankings()

    top_speed = speed_tracker.get_top_speed_player()

    active_player = activity_tracker.get_most_active_player()

    momentum = momentum_tracker.get_momentum()



    # Safety checks

    if top_speed is None:
        top_speed = {
            "player_id": None,
            "speed": 0
        }


    if active_player is None:
        active_player = {
            "player_id": None,
            "score": 0
        }


    if momentum is None:
        momentum = {
            "Dominating": "Unknown"
        }


    if possession is None:
        possession = {
            "Red Team": 0,
            "White Team": 0
        }


    if distance_rank is None:
        distance_rank = []



    analytics = {


        "possession": {

            "Red Team": possession.get(
                "Red Team",
                0
            ),

            "White Team": possession.get(
                "White Team",
                0
            )

        },



        "top_distance_players": [

            {

                "player_id": player[0],

                "distance_meters": round(
                    player[1],
                    2
                )

            }

            for player in distance_rank[:5]

        ],




        "highest_speed": {


            "player_id": top_speed.get(
                "player_id",
                None
            ),


            "speed": top_speed.get(
                "speed",
                top_speed.get(
                    "max_speed",
                    0
                )
            )

        },





        "most_active_player": {


            "player_id": active_player.get(
                "player_id",
                None
            ),


            "activity_score": active_player.get(
                "score",
                active_player.get(
                    "activity_score",
                    0
                )
            )

        },





        "momentum": {


            "dominant_team": momentum.get(
                "Dominating",
                momentum.get(
                    "dominant_team",
                    "Unknown"
                )
            )

        }


    }



    # Create output folder if missing

    os.makedirs(
        "output",
        exist_ok=True
    )



    # Save JSON

    with open(
        "output/analytics.json",
        "w"
    ) as file:


        json.dump(
            analytics,
            file,
            indent=4
        )



    print("analytics.json saved")