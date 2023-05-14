#include <vector>
#include <cstdio>
#include <iostream>
#include <iomanip>

typedef struct {
   size_t ev;			// event number
   size_t po;			// event header position
} FILEINDEX;

int main() {
    FILE *fp;
    int word_offset = 0; // offset in words
    std::uint32_t uiEVENT_HEADER[3] = {0,0,0};
    // Determine the actual number of words in the file

    // Open the binary file for reading
    fp = fopen("main_run_0037_ov_3.00_sipmgr_01_tile_0cd96b60_20230421_175146.data", "rb");
    if (fp == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    fseek(fp,0,SEEK_END);
	  size_t sSizeOfFile = ftell(fp);
    
    // Set the file pointer to the correct byte offset
    fseek(fp, 0 , SEEK_SET);
    int num_events = 0;
    std::uint32_t buffer;
    // Read the binary data into the array
    FILEINDEX FileIndex;
    FileIndex.ev = 0;
    FileIndex.po = 0;
    std::vector<FILEINDEX> vFInd;

    while (!feof(fp)) {
        fread(&buffer, sizeof(std::uint32_t), 1 , fp);
        if (buffer == 0x2A502A50) {
            fseek(fp,-sizeof(uint32_t),SEEK_CUR);
            fread(uiEVENT_HEADER,sizeof(uiEVENT_HEADER),1,fp);
            FileIndex.ev = num_events;   // absolute event number by data file
            FileIndex.po = ftell(fp)-3*sizeof(uint32_t);
            num_events++;
            std::cout << num_events << std::endl;
            vFInd.push_back(FileIndex);
            fseek(fp,FileIndex.po+uiEVENT_HEADER[1],SEEK_SET);		// from zero position - jump over all event length
            if (FileIndex.po+uiEVENT_HEADER[1] > sSizeOfFile) break;
        }
    }
    std::cout << "number of events:" << num_events << std::endl;
    int nev;
    std::cout << "input the event number:";
    std::cin >> nev; 
    std::cout << "input the sync word:";
    std::uint32_t sync_word = 0x70000000;
    int offset = 0;
    std::uint32_t word;
    fseek(fp,vFInd.at(nev).po + sizeof(word) *  offset ,SEEK_SET);
    fread(&word,sizeof(word),1,fp);
    std::cout << offset << "\t" << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
    int pass = 13;
    fseek(fp, sizeof(word) *  pass ,SEEK_CUR);
    offset += pass;
    fread(&word,sizeof(word),1,fp);
    std::cout << offset << "\t" << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
    int chsize =  word & 0x0000FFFF;
    fseek(fp, sizeof(word) *  chsize / 4 ,SEEK_CUR);
    offset += chsize /4;
    fread(&word,sizeof(word),1,fp);
    std::cout << offset << "\t" << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
    // Close the file
    fclose(fp);

    return 0;
}

