#include "dataset.h"
#include <fstream>
#include <dirent.h>

// Define a function to get the list of files or subdirectories in a directory
vector<string> GetEntriesInDirectory(const string& dir_path){
	vector<string> entries;
	// Open the directory
	DIR* dir = opendir(dir_path.c_str());
	if (dir == nullptr)
	{
		std::cerr << "Error: failed to open directory " << dir_path << std::endl;
		return entries;
	}
	// Read the entries in the directory
	struct dirent* entry;
	while ((entry = readdir(dir)) != nullptr)
	{
		// Skip the "." and ".." entries
		if (string(entry->d_name) == "." || string(entry->d_name) == ".."){
			continue;
		}
		// Add the file to the list
		entries.push_back(entry->d_name);
	}
	// Close the directory and return the list of entries
	closedir(dir);
	std::sort(entries.begin(), entries.end());
	return entries;
}

/* 
* @brief: traverse the .jpg file in the folder
* @inputs:
		- the root folder path
		- the lists of images
		- the list of labels
*/
void load_data_from_folder(string folder_path, vector<string> &image_paths, 
	vector<int> &labels)
{
	auto root_dir = folder_path;
	vector<string> sub_dirs = GetEntriesInDirectory(root_dir);
	// Loop over the subdirectories
	for(auto i = 0; i < sub_dirs.size(); i++){
		// Get the list of files in the current subdirectory
		vector<string> files = GetEntriesInDirectory(root_dir + "/" + sub_dirs[i]);
		// Loop over the files and add them to the vectors
		for(string& file : files){
			image_paths.push_back(root_dir + "/" + sub_dirs[i] + "/" + file);
			labels.push_back(i);
		}
	}
}