
define sick_s300 sensor
(				
    # factory settings for Sick S300 sensor
    range [ 0.00 30.0 ]
    pose [ 0 0 0.1 0 ]
    fov 270
    samples 541
    color_rgba [0 0 1 0.15]
)

define sickcommon ranger
(
    color "black"
    size [ 0.102 0.152 0.105 ] # dimensions from Sick S300 data sheet	
)

define sickbase sickcommon
(
    block( points 4
        point[0] [0 0]
        point[1] [0 1]
        point[2] [1 1]
        point[3] [1 0]
        z [0 0.21]
    )
)

# extends sicklaser to add nice-looking but relatively expensive geometry
define fancysickbase sickcommon
(
    # bottom
    block( 
        points 4
        point[0] [ -0.02 -0.077 ]
        point[1] [  0.078 -0.077 ]
        point[2] [  0.078  0.077 ]
        point[3] [ -0.02  0.077 ]
        z [0 0.02 ]
    )

    # back
    block(
        points 4
        point[0] [ -0.078 -0.077 ]
        point[1] [ -0.02  -0.077 ]
        point[2] [ -0.02   0.077 ]
        point[3] [ -0.078  0.077 ]
        z [0 0.21 ]
    )

    # top
    block( points 4
        point[0] [ -0.02 -0.077 ]
        point[1] [  0.078 -0.077 ]
        point[2] [  0.078  0.077 ]
        point[3] [ -0.02  0.077 ]
        z [0.12 0.21 ]  
    )

    # laser bit
    block( points 4
        point[0] [ -0.02 -0.05 ]
        point[1] [  0.06 -0.05 ]
        point[2] [  0.06  0.05 ]
        point[3] [ -0.02  0.05 ]
        z [0.02 0.12 ]  
    )
)

define sick_laser sickbase ( sick_s300() )

define fancysicklaser fancysickbase ( sick_s300() )

