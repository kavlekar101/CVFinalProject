import cv2

# Read the image
image = cv2.imread('edges_of_image.jpg')

# Normalize the image
normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


# Display the image
cv2.imshow('Image', normalized_image)

cv2.imwrite('normalized_image.jpg', normalized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()


