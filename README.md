# Noise_Parquet
The project presents a novel approach of superimposing noise on an image. 

The key words are Poisson disks, pqrquet and minimum boundary cut error.

Initially, blue noise is generated using Poisson disks on a small plane to save time.

Then 4 fragments are selected from the plane that do not overlap. 

Using the 4 fragments, we create a set of tiles from which we will lay the parquet. 
In this way, we obtain a homogeneous blue noise.

Using pixel brightness and the accept-reject method, we apply or do not apply noise to the image, obtaining a non-uniform noise.

In short, this is how the algorithm generates an image with superimposed non-uniform noise.

