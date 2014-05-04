# LinkedList Class, Node Class
class Node:
   def __init__( self, data ):
      self.data = data
      self.next = None

class LinkedList:
   def __init__( self ):
      self.head = None
      self.tail = None

   def add_node( self, data ):
      new_node = Node( data )

      if self.head == None:
         self.head = new_node

      if self.tail != None:
         self.tail.next = new_node

      self.tail = new_node

   def rm_node( self, index ):
      prev = None
      node = self.head
      i = 0

      while ( node != None ) and ( i < index ):
         prev = node
         node = node.next
         i += 1

      if prev == None:
         self.head = node.next
      else:
         prev.next = node.next

   def print_list( self ):
      node = self.head

      while node != None:
         print node.data
         node = node.next

if __name__ == '__main__':

   List = LinkedList()
   List.add_node(1)
   List.add_node(2)
   List.add_node(3)
   List.add_node(4)
   List.print_list( )
   List.rm_node( 2 )
   List.print_list( )

   ## Using python list library
   List = []
   List.append(1)
   List.append(2)
   List.append(3)
   List.append(4)

   for i in List:
      print i
