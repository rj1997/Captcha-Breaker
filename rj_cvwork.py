# This program separates the image into 4 single characters

import cv2
import os

# Path of the image dataset foler
path = "generated_captcha_images" 
counts={}

# Go through each image file
for filename in os.listdir(path):

	fullfilename = path+"/"+filename

	# Read the image into full color
	img_org = cv2.imread(fullfilename,1)

	# Read the image into grayscale
	img = cv2.imread(fullfilename,0)

	# Thresholding the image into black and white scale, color are insignificant in determining the captcha
	ret,thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

	# Find Contours in the image
	im2, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


	Base = os.path.basename(fullfilename)
	fullfilename = os.path.splitext(Base)[0]

	# Save the categorical images in this folder
	OUTPUT_FOLDER = "extracted_letter_images"
	cv2.drawContours(img_org, contours, -1, (0,255,0), 1)

	letter_image_regions = []

	# Let's find individual areas for each image
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		
		# If width is significantly more than the height, there might be two characters overlapping,
		# hence we need to split it in between
		if w/h>1.25:
			new_w=w/2
			letter_image_regions.append((x,y,new_w,h))
			letter_image_regions.append((x+new_w,y,new_w,h))
		else:
			letter_image_regions.append((x,y,w,h))



	# Sort the regions on the basis of their x parameter, so that the first character that comes, comes first in the dictionary
	letter_image_regions.sort(key=lambda x:x[0])

	

	for box,text in zip(letter_image_regions,fullfilename):
		x,y,w,h = box
		# y coordinate is transitioned in the number of rows
		# x coordinate is transitioned in the number of columns
		individual_image = img[y-1:y+h+1,x-1:x+w+1]


		save_path = os.path.join(OUTPUT_FOLDER,text)
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		count = counts.get(text, 1)

		# The final path of the image
		p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))

		# Write the image into the memory
		cv2.imwrite(p, individual_image)
		counts[text] = count + 1

