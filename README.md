# Facial-Feature-Recognition-and-Alignment-Program

## Summary 
  Our project will consist of a program which can take a series of photos of different faces and will output a video collage progressing through these photos with all the faces aligned. We plan on allowing the user to select for which facial feature they want to align on, whether it be the eyes, nose, mouth, etc. The project will mainly focus on being able to recognize different facial features, and how to normalize this information among many different photos, which may have different conditions we have to account for.

## Background
  We will be doing some facial mapping similar to what is seen with AI deepfakes. However, instead of mapping one face onto another, we will use these points to align one still picture with another.

## The Challenge 
  Our project will have 2 main challenges. The first of these challenges will be the object recognition involved in determining facial features. The second challenge will be calculating and executing the necessary transformation on each image to align them since each photo will be of a different person. While opencv does provide methods for object detection, we will need to train it to detect the facial features we are interested in. Other challenges include how it will handle 2 faces in 1 photo, how we will need to preserve the proper proportion of the face while aligning it, and dealing with photos where a face can’t be detected.

## Goals and Deliverables
1. to be able to detect facial features from a collection of different photos, rejecting the photos where these facial features do not appear
2. For each of these photos, use the data we get from the facial feature detector to center all of the faces, scale/crop the photos so they are of approximately equal size and rotate if needed to make sure the faces are level
3. Export our edited photos into a format which will allow the user to see the aligned images displayed in quick succession

Additionally, if time allows, we would like to implement a way to cycle through the photos in an organized way depending on a categorization given to each photo. For example, having the output show the progression from the lightest blue eyes to the darkest brown eyes. 
	The project’s success can be determined by observing the output. If the facial feature in each photo is properly aligned without issue, then the project can be deemed successful.
	Realistically, the project should be able to be completed in the allotted time and there is a decent chance that we will have enough time remaining to implement the additional features

## Schedule 

| Week #  | Alex | Wil |
|---------|-----------|-----------------|
|Week One | Get familiar with OpenCV and its functions | Learning OpenCV |
|Week Two | Develop data scraper to obtain photos | Begin work on training to detect faces |
|Week Three | Get program to take photos as input and output a slideshow of the photos inputted | Becoming familiar with the different ways we can transform images to our liking |
|Week Four | Work on facial feature recognition | Work on facial feature recognition |
|Week Five | Work on aligning facial features | Work on aligning facial features |
|Week Six | Research valid photo manipulations | Begin testing our additional goal of categorizing the photos based on different features |
|Week Seven | Implement photo manipulation | If time allows it, finishing the additional categorization. If time could be better spent finishing the main program then I will do that |
|Week Eight | Testing and debugging | Testing and debugging |
|Week Nine | Testing and debugging. If time allows, work on additional categorization features if time allows | Testing and debugging. If time allows, work on additional categorization features if time allows | 














