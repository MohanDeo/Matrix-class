#include<iostream>
#include<sstream>
#include<cmath>
#include<string.h>
#include<vector>
#include<numeric>

class matrix
{
friend std::ostream & operator<<(std::ostream &os, const matrix &m);
friend std::istream & operator>>(std::istream &is, const matrix &m);

private:
  int number_of_columns;
  int number_of_rows;
  double *matrix_data;

public:
  matrix(){
    matrix_data = nullptr;
    number_of_columns=number_of_rows=0;}
    
  matrix(int number_of_row_input, int number_of_col_input, double matrix_values[]):
  number_of_rows{number_of_row_input}, number_of_columns{number_of_col_input} {
      matrix_data = new double[number_of_row_input*number_of_col_input];
      for(int i=0; i<number_of_row_input*number_of_col_input; ++i){
          matrix_data[i]=matrix_values[i];
      }
  }

  ~matrix(){
    //Only print when destuctor used on filled matrix
    if(matrix_data !=nullptr){
        std::cout<<"Destroying "<<number_of_rows<<" x "<<number_of_columns<<" matrix"<<std::endl;
    }

    delete[] this->matrix_data;
    
  }



  double row_return_func(){
      return number_of_rows;
  }

  double col_return_func(){
      return number_of_columns;
  }

  matrix delete_row_col(){
      std::string col_delete;
      std::cout<<"Would you like to delete a column (y/n)?"<<std::endl;
      std::cin>>col_delete;
      if(col_delete=="y"){
        int col_specifier;
        std::cout<<"What column would you like to delete? Please enter integer(1 to "
        <<number_of_columns<<")"<<std::endl;
        std::cin>>col_specifier;
        std::vector<double> spliced_matrix_vec;
        for(int i{0}; i<number_of_rows; i++){
            for(int j{0}; j<number_of_columns; j++){
                if(j==(col_specifier-1)){
                    ++j;
                }
                spliced_matrix_vec.push_back(matrix_data[j + i*number_of_columns]);
            }
        }
        double new_matrix_data[(number_of_columns - 1) * number_of_rows];
        std::vector<double>::iterator vector_begin{spliced_matrix_vec.begin()};
        std::vector<double>::iterator vector_end{spliced_matrix_vec.end()};
        std::vector<double>::iterator vector_iterator;
        int element{0};
        for(vector_iterator=vector_begin;vector_iterator<vector_end;++vector_iterator){
            new_matrix_data[element] = *vector_iterator;
            element++;
        }
        std::vector<double>().swap(spliced_matrix_vec);
        matrix new_matrix{number_of_rows, number_of_columns - 1, new_matrix_data};
        std::string row_delete;
        std::cout<<"Would you like to delete a row (y/n)?"<<std::endl;
        std::cin>>row_delete;
        if(row_delete=="y"){
            int row_specifier;
            std::cout<<"What row would you like to delete? Please enter integer(1 to "<<number_of_rows
            <<")"<<std::endl;
            std::cin>>row_specifier;
            for(int i{0}; i<number_of_rows; i++){
                if(i==(row_specifier-1)){
                    ++i;
                }
                for(int j{0}; j<new_matrix.number_of_columns; j++){
                    spliced_matrix_vec.push_back(new_matrix_data[j + i*new_matrix.number_of_columns]);
                }
            }
            double new_matrix_data_1[new_matrix.number_of_rows * (number_of_rows - 1)];
            std::vector<double>::iterator vector_begin{spliced_matrix_vec.begin()};
            std::vector<double>::iterator vector_end{spliced_matrix_vec.end()};
            std::vector<double>::iterator vector_iterator;
            int element_1{0};
            for(vector_iterator=vector_begin;vector_iterator<vector_end;++vector_iterator){
                new_matrix_data_1[element_1] = *vector_iterator;
                element_1++;
            }
            matrix new_matrix_1{number_of_rows - 1, new_matrix.number_of_columns, new_matrix_data_1};
            return new_matrix_1;
        }
        return new_matrix;
    }else if(col_delete == "n"){
                std::string row_delete;
        std::cout<<"Would you like to delete a row (y/n)?"<<std::endl;
        std::cin>>row_delete;
        if(row_delete=="y"){
            std::vector<double> spliced_matrix_vec;
            int row_specifier;
            std::cout<<"What row would you like to delete? Please enter integer(1 to "<<
            number_of_rows<<")"<<std::endl;
            std::cin>>row_specifier;
            for(int i{0}; i<number_of_rows; i++){
                if(i==(row_specifier-1)){
                    ++i;
                }
                for(int j{0}; j<number_of_columns; j++){
                    spliced_matrix_vec.push_back(matrix_data[j + i*number_of_columns]);
                }
            }
            double new_matrix_data_1[number_of_rows * (number_of_rows - 1)];
            std::vector<double>::iterator vector_begin{spliced_matrix_vec.begin()};
            std::vector<double>::iterator vector_end{spliced_matrix_vec.end()};
            std::vector<double>::iterator vector_iterator;
            int element_1{0};
            for(vector_iterator=vector_begin;vector_iterator<vector_end;++vector_iterator){
                new_matrix_data_1[element_1] = *vector_iterator;
                element_1++;
            }
            matrix new_matrix_1{number_of_rows - 1, number_of_columns, new_matrix_data_1};
            return new_matrix_1;
        }else if(row_delete=="n"){
            matrix orig_matrix{number_of_rows, number_of_columns, matrix_data};
            return orig_matrix;
        }
    }
    std::cout<<"Your input was not recognised"<<std::endl;
    matrix empty_matrix;
    return empty_matrix;
}

  // Overload + operator for addition
  matrix operator +(const matrix &m) const
  {
      int orig_row_no{number_of_rows};
      int orig_col_no{number_of_columns};

      if((m.number_of_rows-orig_row_no)!=0 || (m.number_of_columns-orig_col_no)!=0){
          std::cout<<"This operation is not possible."<<std::endl;
          matrix empty;
          return empty;
      }
    double new_matrix_array[m.number_of_rows*m.number_of_columns];
    for(int i{0}; i<m.number_of_rows*number_of_columns; ++i){
        new_matrix_array[i] = matrix_data[i] + m.matrix_data[i];
    }
    matrix new_matrix{m.number_of_rows, m.number_of_columns, new_matrix_array};
    return new_matrix;
  }

  // Overload - operator for addition
  matrix operator -(const matrix &m) const
  {
      int orig_row_no{number_of_rows};
      int orig_col_no{number_of_columns};

      if((m.number_of_rows-orig_row_no)!=0 || (m.number_of_columns-orig_col_no)!=0){
          std::cout<<"This operation is not possible."<<std::endl;
          matrix empty;
          return empty;
      }
    double new_matrix_array[m.number_of_rows*m.number_of_columns];
    for(int i{0}; i<m.number_of_rows*number_of_columns; ++i){
        new_matrix_array[i] = matrix_data[i] - m.matrix_data[i];
    }
    matrix new_matrix{m.number_of_rows, m.number_of_columns, new_matrix_array};
    return new_matrix;
  }

  matrix operator *(const matrix &m) const
   {
       if(number_of_columns!=m.number_of_rows){
           std::cout<<"This operation is not possible."<<std::endl;
          matrix empty;
          return empty; 
       }
       //Making array of rows for first matrix
       double spliced_matrix_array_rows[number_of_rows][number_of_columns];
       for(int i{0}; i<number_of_rows; i++){
           for(int j{0}; j<number_of_columns; j++){
               spliced_matrix_array_rows[i][j] = matrix_data[j + i*number_of_columns];
            }
        }

        //Making array of cols for second matrix
       double spliced_matrix_array_cols[number_of_columns][m.number_of_rows];
       for(int i{0}; i<m.number_of_columns; i++){
           for(int j{0}; j<m.number_of_rows; ++j){
               spliced_matrix_array_cols[i][j] = m.matrix_data[j*m.number_of_columns + i];
            }
        }

    double *new_array_data;
    new_array_data = new double[number_of_rows * m.number_of_columns];

        //Multiply each row by all columns
        for(int i{0}; i<number_of_rows; i++){
            for(int j{0}; j<m.number_of_columns; j++){
                double temp_array[number_of_columns];
                for(int k{0}; k<number_of_columns; ++k){
                    temp_array[k] = spliced_matrix_array_rows[i][k] * spliced_matrix_array_cols[j][k];
                }
                double sum{0};
                for(auto& num : temp_array){
                    sum += num;
                }
                new_array_data[j + i * m.number_of_columns] = sum;
            }
        }
        matrix new_matrix{number_of_rows, m.number_of_columns, new_array_data};
        return new_matrix;
    }


//Determinant of a matrix
 double get_det(){
    #define N number_of_rows

    if(number_of_rows!=number_of_columns){
        std::cout<<"This matrix is not square."<<std::endl;
        return 0.0;
    }
    std::vector<std::vector<double>>mat;
       for(int i{0}; i<number_of_rows; i++){
           std::vector<double>indexes;
           for(int j{0}; j<number_of_columns; j++){
               indexes.push_back(matrix_data[j + i*number_of_columns]);
            }
            mat.push_back(indexes);
            std::vector<double>().swap(indexes);
        }
    return determinant_function(mat, number_of_rows);
}

//Function to get cofactor of mat[cofactor_row][cofactor_col] in temp[][]
std::vector<std::vector<double>>  cofactor_function(std::vector<std::vector<double>>mat, 
std::vector<std::vector<double>>temp, int cofactor_row, int cofactor_col, int current_dimension){
	
    int i = 0, j = 0;
    
	for (int row{0}; row < current_dimension; row++){
		for (int col{0}; col < current_dimension; col++){
			//Making a cofactor matrix
			if (row != cofactor_row && col != cofactor_col){
                temp[i].push_back(mat[row][col]);
                j++;
				//All cols got from row, so move to next row
				if(j == (current_dimension - 1)){
					j = 0;
					i++;
				}
			}
		}
	}
    return temp;
}

// Recursive function for finding determinant of matrix.
double determinant_function(std::vector<std::vector<double>>mat, int current_dimension){

	double det = 0.0;

	// Base case if matrix contains single element and atomic case if 2x2
	if (current_dimension == 1){
		return mat[0][0];
    } else if(current_dimension == 2){
        det += (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]);
        return det;
    }

    // To store cofactors
    std::vector<std::vector<double>>temp;
    temp.resize(current_dimension);

	double sign = 1.0; //Cofactor multiplier sign
    std::vector<std::vector<double>> temp_1;

	for (int cofactor_multiplier_index{0}; cofactor_multiplier_index < current_dimension; 
    cofactor_multiplier_index++){
		//Getting Cofactor of mat[0][cofactor_multiplier_index]
        //Cofactor row is always 0
		temp_1 = cofactor_function(mat, temp, 0, cofactor_multiplier_index, current_dimension);
		det += sign * mat[0][cofactor_multiplier_index]
			* determinant_function(temp_1, (current_dimension - 1));
		//Cofactor multipliers alternate sign
		sign = -sign;
	}
	return det;
}

//Move constructor
matrix(matrix&& mat):matrix_data(std::move(mat.matrix_data)){
    number_of_rows = mat.number_of_rows;
    number_of_columns = mat.number_of_columns;
    mat.number_of_columns = 0;
    mat.number_of_rows = 0;
    mat.matrix_data = nullptr;
}

//Copy constructor using deep copying
matrix(const matrix &to_copy):matrix{to_copy.number_of_rows, to_copy.number_of_columns, 
to_copy.matrix_data} {
}

//Move assignment operator
matrix & operator=(matrix&& rhs){

  std::cout <<"move assignment\n";
  std::swap(number_of_rows,rhs.number_of_rows);
  std::swap(number_of_columns,rhs.number_of_columns);
  std::swap(matrix_data,rhs.matrix_data);
  return *this;
}

// Assignment operator using deep copying
matrix& operator=(matrix &rhs){
  std::cout <<"copy assignment\n"<<std::endl;;
  if(&rhs == this){
      return *this; 
   }// no self assignment
  delete[] matrix_data; 
  matrix_data=nullptr;
  number_of_rows = 0;
  number_of_columns = 0;
  number_of_rows = rhs.number_of_rows;
  number_of_columns = rhs.number_of_columns;
  if(number_of_rows*number_of_columns>0){
      matrix_data=new double[number_of_rows*number_of_columns];
      for(int i{};i<number_of_rows*number_of_columns;i++){
          matrix_data[i] = rhs.matrix_data[i];
        }
    }
  return *this;
}

};
// Function to overload << operator
std::ostream & operator<<(std::ostream &os, const matrix &m) {
    std::cout<<"The "<<m.number_of_rows<<" x "<<m.number_of_columns<<" matrix:\n"<<std::endl;
    int i{0};
    double new_line;
    for(int j{0}; j<m.number_of_rows; ++j) {
        for(i; i<m.number_of_columns*m.number_of_rows; ++i){
            std::cout<<" "<<m.matrix_data[i]<<" ";
            new_line = (i+1.0)/m.number_of_columns;
            if(std::abs(new_line-std::round(new_line))==0){
                std::cout<<std::endl;
            }
        }
    }
  return os;
}

//Function to overload >> operator
std::istream & operator>>(std::istream &os, matrix &m){
  int no_cols;
  int no_rows;
  std::string matrix_string;
  double matrix_elements;
  std::cout<<"What is the number of rows?"<<std::endl;
  os>>no_rows;
  std::cout<<"What is the number of columns?"<<std::endl;
  os>>no_cols;
  double matrix_array[no_rows*no_cols];

  std::cout<<"Please enter the values in the matrix from left to right as a,b,c,d with "
  "no spaces"<<std::endl;
  os>>matrix_string;
 
  std::stringstream ss(matrix_string);
  std::vector<double> matrix_elements_vector;

    while(ss.good()){
      std::string matrix_elements_string;
      getline(ss, matrix_elements_string, ',');
      matrix_elements = atof(matrix_elements_string.c_str());
      matrix_elements_vector.push_back(matrix_elements);
      
  }

  double* new_data = new double[matrix_elements_vector.size()];
  std::copy(matrix_elements_vector.begin(), matrix_elements_vector.end(), new_data);

  double* mat_dat{&new_data[0]};

  for(int i{0}; i<no_rows*no_cols; ++i){
        matrix_array[i] = mat_dat[i];
    }

    const matrix temp{no_rows, no_cols, matrix_array};
    matrix new_matrix{temp};
    m = new_matrix;
    
    return os;
}

int main()
{
    std::cout.precision(3);

    //Parameterised constructor
    double hi_1[9]{5,5,4,1,2,3,6,9,8};
    matrix B{3, 3, hi_1};
    double hi_2[6]{3,4,1,2,5,6};
    matrix C{2, 3, hi_2};
    
    //Input a matrix
    matrix A;
    std::cout<<"Enter your matrix"<<std::endl;
    std::cin>>A;
    std::cout<<A<<std::endl;
    
    //Addition and subtraction
    std::cout<<A-B<<std::endl;
    std::cout<<C-A<<std::endl;//Not possible
    std::cout<<A+B<<std::endl;
    std::cout<<C+A<<std::endl;//Not possible

    //Multiplication
    std::cout<<A*B<<std::endl;
    std::cout<<C*A<<std::endl;
    std::cout<<B*C<<std::endl;//Not possible

    //Recursive determinant
    std::cout<<A.get_det()<<std::endl;
    std::cout<<C.get_det()<<std::endl;//Not possible

    //Delete i and j columns
    matrix V{B.delete_row_col()};
    std::cout<<V<<std::endl;

    //Move and copy
    matrix L;
    L=A;

    std::cout<<L<<std::endl;
    std::cout<<A<<std::endl;
    matrix H;
    H = std::move(A);
    std::cout<<A.row_return_func()<<std::endl;
    std::cout<<A.col_return_func()<<std::endl;
    std::cout<<H.row_return_func()<<std::endl;
    std::cout<<H.col_return_func()<<std::endl;
    std::cout<<L<<std::endl;//L is the same even though A is modified

    return 0;
}
