import json
import math



def load_json(path):

    """
    Load detections.json file
    """

    with open(path, "r") as file:

        data = json.load(file)

    return data




def calculate_distance(x1, y1, x2, y2):

    """
    Euclidean distance calculation
    """

    distance = math.sqrt(

        (x2 - x1) ** 2 +

        (y2 - y1) ** 2

    )

    return distance




def group_by_frame(detections):

    """
    Convert flat JSON into frame-wise groups

    Input:

    [
      {frame:0, player_id:1},
      {frame:0, player_id:2},
      {frame:1, player_id:1}
    ]


    Output:

    {
      0:[
          player1,
          player2
        ],

      1:[
          player1
        ]
    }

    """


    frames = {}


    for obj in detections:


        frame_number = obj["frame"]


        if frame_number not in frames:

            frames[frame_number] = []


        frames[frame_number].append(obj)



    return frames