/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./threads/AccessorContainer.h

    Copyright (C) 2015

Author: Gianluca Filaci <g.filaci@ed.ac.uk>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */

class AccessorListElementBase {
 public:
  virtual void require(cl::sycl::handler &cgh) {};
  virtual void create_host_accessor() {};
  virtual void delete_host_accessor() {};
  virtual ~AccessorListElementBase() = default;
};

template <class T>
class AccessorListElement : public AccessorListElementBase {
 public:
  T* acc_ptr;
  void require(cl::sycl::handler &cgh) { cgh.require(acc_ptr->da_ptr->sycl_da); }
  void create_host_accessor() { acc_ptr->create_host_accessor(); }
  void delete_host_accessor() { acc_ptr->delete_host_accessor(); }
 AccessorListElement(T* acc_ptr_) : acc_ptr(acc_ptr_) {}
};

class AccessorContainer {
 public:
  std::vector<AccessorListElementBase*> list;

    template<class T>
      void push_back(T* acc_ptr) {
      list.push_back(new AccessorListElement<T>(acc_ptr));
    }
    void require(cl::sycl::handler &cgh) {
      for(int i=0; i<list.size(); i++) list[i]->require(cgh);
    }
    void create_host_accessors() {
      for(int i=0; i<list.size(); i++) list[i]->create_host_accessor();
    }
    void delete_host_accessors() {
      for(int i=0; i<list.size(); i++) list[i]->delete_host_accessor();
    }
    void clear() {
      for(int i=0; i<list.size(); i++) delete list[i];
      list.clear();
    }
};
